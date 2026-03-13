"""Vertical Federated Learning (VFL) — Split Learning variant.

In VFL every client owns a **disjoint subset of features** for the *same* set of samples.
Each client trains a *bottom model* that maps its local features to an embedding.
The server holds the labels and a *top model* that takes the concatenated embeddings from
all clients and produces predictions.

Training protocol (per round):
    1. Server sends the current top-model to nobody (it stays on the server).
    2. Each client does a forward pass through its bottom model → embedding.
    3. Embeddings are sent to the server.
    4. Server concatenates embeddings, does forward + loss + backward through the top model.
    5. Server sends the gradient w.r.t. each client's embedding back to that client.
    6. Each client backpropagates the received gradient through its bottom model and updates.
    7. Server updates its own top model.

References:
    - Vepakomma et al. "Split learning for health: Distributed deep learning without
      sharing raw data." (2018) arXiv:1812.00564
"""

from __future__ import annotations

import sys
import uuid
from copy import deepcopy
from typing import Any, Callable, Collection, Sequence

import numpy as np
import torch

sys.path.append(".")
sys.path.append("..")

from .. import DDict, FlukeENV, ObserverSubject  # NOQA
from ..comm import Channel, ChannelObserver, Message  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..data.vertical import VerticalDataSplitter  # NOQA
from ..evaluation import Evaluator  # NOQA
from ..utils import ClientObserver, ServerObserver, get_loss, get_model  # NOQA
from ..utils.model import ModOpt  # NOQA

__all__ = ["VerticalFL"]


class _VFLInferenceModel(torch.nn.Module):
    """Inference-only wrapper to evaluate VFL with Fluke's evaluator.

    It reconstructs end-to-end predictions from:
      - client bottom models (feature-wise)
      - server top model (on concatenated embeddings)
    """

    def __init__(
        self,
        bottom_models: Sequence[torch.nn.Module],
        top_model: torch.nn.Module,
        feature_splits: list[np.ndarray],
    ) -> None:
        super().__init__()
        self.bottom_models = torch.nn.ModuleList(bottom_models)
        self.top_model = top_model
        self.feature_splits = [torch.as_tensor(fs, dtype=torch.long) for fs in feature_splits]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for bottom, feat_idx in zip(self.bottom_models, self.feature_splits):
            X_client = X.index_select(dim=1, index=feat_idx.to(X.device))
            embeddings.append(bottom(X_client))
        concat = torch.cat(embeddings, dim=1)
        return self.top_model(concat)


# ──────────────────────────────────────────────────────────────────────────────
#  VFL Client
# ──────────────────────────────────────────────────────────────────────────────


class VFLClient:
    """Client in a Vertical FL setting.

    Each client holds a *bottom model* that maps its local feature slice to an embedding
    vector.  During training the client:

    1. Computes the forward pass (features → embedding) and sends the embedding to the
       server.
    2. Receives the gradient of the loss w.r.t. its embedding from the server.
    3. Back-propagates that gradient through the bottom model and updates weights.

    Args:
        index (int): Client identifier.
        train_set (FastDataLoader): Local training features (no labels).
        test_set (FastDataLoader): Local test features (no labels).
        bottom_model (torch.nn.Module): The bottom model for this client.
        optimizer_cfg (OptimizerConfigurator): Optimizer configurator.
        local_epochs (int): Number of local epochs per round (usually 1 in split learning).
    """

    def __init__(
        self,
        index: int,
        train_set: FastDataLoader,
        test_set: FastDataLoader,
        bottom_model: torch.nn.Module,
        optimizer_cfg: OptimizerConfigurator,
        local_epochs: int = 1,
        **kwargs,
    ):
        self._index: int = index
        self.train_set: FastDataLoader = train_set
        self.test_set: FastDataLoader = test_set
        self.device: torch.device = FlukeENV().get_device()

        self.bottom_model: torch.nn.Module = bottom_model
        self._optimizer_cfg = optimizer_cfg
        self.optimizer, self.scheduler = self._optimizer_cfg(self.bottom_model)
        self.local_epochs: int = local_epochs

        self._channel: Channel | None = None
        self._last_round: int = 0

    @property
    def index(self) -> int:
        return self._index

    @property
    def n_examples(self) -> int:
        if isinstance(self.train_set, FastDataLoader):
            return self.train_set.size
        return len(self.train_set.dataset)

    def set_channel(self, channel: Channel) -> None:
        self._channel = channel

    @property
    def channel(self) -> Channel:
        return self._channel

    # ── forward pass ──────────────────────────────────────────────────────

    def compute_embeddings(self, X_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through the bottom model.  Returns a tensor with
        ``requires_grad=True`` so that the server's gradient can flow back."""
        self.bottom_model.train()
        self.bottom_model.to(self.device)
        X_batch = X_batch.to(self.device)
        embedding = self.bottom_model(X_batch)
        return embedding

    # ── backward pass ─────────────────────────────────────────────────────

    def backward_step(self, grad_embedding: torch.Tensor) -> None:
        """Receive the gradient w.r.t. the embedding, backpropagate, and update."""
        self.optimizer.zero_grad()
        self._cached_embedding.backward(grad_embedding.to(self.device))
        torch.nn.utils.clip_grad_norm_(self.bottom_model.parameters(), max_norm=1.0)
        self.optimizer.step()

    def __str__(self) -> str:
        return f"VFLClient[{self._index}]"

    def __repr__(self) -> str:
        return self.__str__()


# ──────────────────────────────────────────────────────────────────────────────
#  VFL Server
# ──────────────────────────────────────────────────────────────────────────────


class VFLServer:
    """Server in a Vertical FL setting.

    The server holds the labels and a *top model* that maps concatenated client embeddings
    to predictions.

    Args:
        top_model (torch.nn.Module): The server-side top model.
        loss_fn (torch.nn.Module): Loss function.
        train_labels (FastDataLoader): Training labels aligned with the client data.
        test_set (FastDataLoader | None): Full test set ``(X, y)`` for global evaluation.
        clients (Sequence[VFLClient]): The VFL clients.
        optimizer_cfg (OptimizerConfigurator): Optimizer configurator for the top model.
        feature_splits (list[np.ndarray]): Feature index arrays per client (needed for eval).
    """

    def __init__(
        self,
        top_model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        train_labels: FastDataLoader,
        test_set: FastDataLoader | None,
        clients: Sequence[VFLClient],
        optimizer_cfg: OptimizerConfigurator,
        feature_splits: list[np.ndarray],
        **kwargs,
    ):
        self.device: torch.device = FlukeENV().get_device()
        self.top_model: torch.nn.Module = top_model
        self.loss_fn = loss_fn
        self.train_labels: FastDataLoader = train_labels
        self.test_set: FastDataLoader = test_set
        self.clients = clients
        self.n_clients = len(clients)
        self.rounds: int = 0
        self.feature_splits = feature_splits

        self._optimizer_cfg = optimizer_cfg
        self.optimizer, self.scheduler = self._optimizer_cfg(self.top_model)

        self._channel: Channel = Channel()
        self._coordinator = ObserverSubject()

    @property
    def channel(self) -> Channel:
        return self._channel

    # ── single training step on one batch ────────────────────────────────

    def train_step(
        self, embeddings: list[torch.Tensor], y_batch: torch.Tensor
    ) -> tuple[float, list[torch.Tensor]]:
        """Perform forward + backward on the top model.

        Args:
            embeddings: list of tensors (one per client), each with ``requires_grad=True``.
            y_batch: ground-truth labels for this batch.

        Returns:
            loss value (float) and list of gradients w.r.t. each client embedding.
        """
        self.top_model.train()
        self.top_model.to(self.device)

        # Concatenate embeddings along feature dimension
        concat = torch.cat(embeddings, dim=1).to(self.device)
        concat.retain_grad()  # needed because concat is not a leaf tensor
        y_batch = y_batch.to(self.device)

        self.optimizer.zero_grad()
        logits = self.top_model(concat)
        loss = self.loss_fn(logits, y_batch)
        loss.backward()

        # Collect gradient for each client's embedding slice
        grads = []
        offset = 0
        for emb in embeddings:
            emb_dim = emb.shape[1]
            grad_slice = concat.grad[:, offset: offset + emb_dim].clone()
            grads.append(grad_slice)
            offset += emb_dim

        torch.nn.utils.clip_grad_norm_(self.top_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item(), grads

    # ── global evaluation using bottom + top models end-to-end ───────────

    @torch.no_grad()
    def evaluate(self, evaluator: Evaluator, round: int) -> dict[str, float]:
        """Evaluate end-to-end on the global test set.

        For each test batch we run each client's bottom model on its features,
        concatenate the embeddings, then run the top model.
        """
        if self.test_set is None:
            return {}

        eval_model = _VFLInferenceModel(
            bottom_models=[client.bottom_model for client in self.clients],
            top_model=self.top_model,
            feature_splits=self.feature_splits,
        )

        return evaluator.evaluate(
            round=round,
            model=eval_model,
            eval_data_loader=self.test_set,
            loss_fn=self.loss_fn,
            device=self.device,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  VFL Algorithm Orchestrator
# ──────────────────────────────────────────────────────────────────────────────


class VerticalFL:
    """Orchestrator for Vertical Federated Learning using Split Learning.

    This class handles:
      - Vertical data splitting across clients (by features).
      - Instantiation of VFL clients (each with a bottom model).
      - Instantiation of the VFL server (with a top model and labels).
      - The training loop: aligned mini-batch forward/backward across clients and server.

    Args:
        n_clients (int): Number of clients.
        data_splitter (VerticalDataSplitter): The vertical data splitter.
        hyper_params (DDict): Hyperparameters including client, server, model info.
    """

    def __init__(
        self,
        n_clients: int,
        data_splitter: VerticalDataSplitter,
        hyper_params: DDict | dict[str, Any],
        **kwargs,
    ):
        if isinstance(hyper_params, dict):
            hyper_params = DDict(hyper_params)

        self._id = str(uuid.uuid4().hex)
        FlukeENV().open_cache(self._id)

        self.hyper_params = hyper_params
        self.n_clients = n_clients
        self.rounds = 0

        # ── split data vertically ─────────────────────────────────────────
        split = data_splitter.assign(n_clients, hyper_params.client.batch_size)
        clients_train = split["clients_train"]
        clients_test = split["clients_test"]
        server_train = split["server_train"]
        server_test = split["server_test"]
        feature_splits = split["feature_splits"]

        # ── build bottom models (one per client) ──────────────────────────
        embedding_dim = (
            hyper_params.net_args.embedding_dim
            if "net_args" in hyper_params and "embedding_dim" in hyper_params.net_args
            else 32
        )
        hidden_dim = (
            hyper_params.net_args.hidden_dim
            if "net_args" in hyper_params and "hidden_dim" in hyper_params.net_args
            else 64
        )
        output_dim = (
            hyper_params.net_args.output_dim
            if "net_args" in hyper_params and "output_dim" in hyper_params.net_args
            else data_splitter.num_classes
        )

        BottomModel = get_model(
            hyper_params.model_bottom,
            input_dim=len(feature_splits[0]),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ) if isinstance(hyper_params.model_bottom, str) else hyper_params.model_bottom

        # ── build clients ─────────────────────────────────────────────────
        client_opt_cfg = OptimizerConfigurator(
            optimizer_cfg=hyper_params.client.optimizer,
            scheduler_cfg=hyper_params.client.scheduler,
        )
        loss_fn = get_loss(hyper_params.client.loss)

        self.clients: list[VFLClient] = []
        for i in range(n_clients):
            # Each client may have a different number of features
            n_feat = len(feature_splits[i])
            bottom = get_model(
                hyper_params.model_bottom,
                input_dim=n_feat,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
            ) if isinstance(hyper_params.model_bottom, str) else deepcopy(hyper_params.model_bottom)

            client = VFLClient(
                index=i,
                train_set=clients_train[i],
                test_set=clients_test[i],
                bottom_model=bottom,
                optimizer_cfg=client_opt_cfg,
                local_epochs=hyper_params.client.local_epochs,
            )
            self.clients.append(client)

        # ── build top model + server ──────────────────────────────────────
        top_input_dim = embedding_dim * n_clients
        top_model = get_model(
            hyper_params.model_top,
            input_dim=top_input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        ) if isinstance(hyper_params.model_top, str) else hyper_params.model_top

        server_opt_cfg = OptimizerConfigurator(
            optimizer_cfg=hyper_params.server.optimizer,
            scheduler_cfg=hyper_params.server.scheduler,
        )

        self.server = VFLServer(
            top_model=top_model,
            loss_fn=deepcopy(loss_fn),
            train_labels=server_train,
            test_set=server_test,
            clients=self.clients,
            optimizer_cfg=server_opt_cfg,
            feature_splits=feature_splits,
        )

        self._channel = Channel()
        for client in self.clients:
            client.set_channel(self._channel)

        self._coordinator = ObserverSubject()
        self.feature_splits = feature_splits

    # ── callbacks ─────────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        return str(self._id)

    def set_callbacks(self, callbacks: Callable | Collection[Callable]) -> None:
        if not isinstance(callbacks, Collection):
            callbacks = [callbacks]
        self._coordinator.attach([c for c in callbacks if isinstance(c, ServerObserver)])
        self._channel.attach([c for c in callbacks if isinstance(c, ChannelObserver)])

    # ── training loop ─────────────────────────────────────────────────────

    def run(self, n_rounds: int, eligible_perc: float = 1.0, finalize: bool = True, **kwargs):
        """Run the VFL training for *n_rounds*.

        One "round" in VFL consists of iterating over the entire training set by aligned
        mini-batches.  Within each mini-batch:
        1. Every client computes its embedding (forward through bottom model).
        2. Embeddings are sent to the server.
        3. Server computes loss + backward through the top model.
        4. Server sends gradient slices back to each client.
        5. Each client back-propagates through its bottom model.
        """
        with FlukeENV().get_live_renderer():
            progress_fl = FlukeENV().get_progress_bar("FL")
            task_rounds = progress_fl.add_task("[red]VFL Rounds", total=n_rounds)

            total_rounds = self.rounds + n_rounds
            for rnd in range(self.rounds, total_rounds):
                self._coordinator.notify(event="start_round", round=rnd + 1, global_model=None)

                epoch_loss = self._train_one_round(rnd + 1)

                # ── evaluation ────────────────────────────────────────────
                if FlukeENV().get_eval_cfg().server:
                    evaluator = FlukeENV().get_evaluator()
                    evals = self.server.evaluate(evaluator, round=rnd + 1)
                    if evals:
                        self._coordinator.notify(
                            event="server_evaluation",
                            round=rnd + 1,
                            eval_type="global",
                            evals=evals,
                        )

                self._coordinator.notify(event="end_round", round=rnd + 1)
                self.rounds += 1
                progress_fl.update(task_id=task_rounds, advance=1)

            progress_fl.remove_task(task_rounds)

        self._coordinator.notify(event="finished", round=self.rounds + 1)

    def _train_one_round(self, current_round: int) -> float:
        """Train for one full pass over the training data (one epoch = one round)."""
        device = self.server.device

        # Reset iterators — all loaders must yield batches in the same order
        client_iters = [iter(client.train_set) for client in self.clients]
        label_iter = iter(self.server.train_labels)

        running_loss = 0.0
        n_batches = 0
        local_epochs = self.clients[0].local_epochs

        for _ in range(local_epochs):
            client_iters = [iter(client.train_set) for client in self.clients]
            label_iter = iter(self.server.train_labels)

            for batch_data in zip(*client_iters, label_iter):
                # Last element is (y,) from server labels
                y_batch = batch_data[-1]
                if isinstance(y_batch, (tuple, list)):
                    y_batch = y_batch[0]

                # Client forward passes
                embeddings = []
                for i, client in enumerate(self.clients):
                    X_client = batch_data[i]
                    if isinstance(X_client, (tuple, list)):
                        X_client = X_client[0]

                    emb = client.compute_embeddings(X_client)
                    # Cache the embedding for backward pass
                    client._cached_embedding = emb
                    # Detach and re-require grad so server can compute gradients
                    emb_detached = emb.detach().requires_grad_(True)
                    embeddings.append(emb_detached)

                    # Track uplink communication (client -> server): embedding tensor
                    self._channel.notify(
                        event="message_received",
                        by=self.server,
                        message=Message(
                            payload=emb_detached.detach(),
                            msg_type="vfl_embedding",
                            sender=client,
                        ),
                    )

                # Server forward + backward
                loss_val, grads = self.server.train_step(embeddings, y_batch)
                running_loss += loss_val
                n_batches += 1

                # Client backward passes
                for i, client in enumerate(self.clients):
                    # Track downlink communication (server -> client): gradient tensor
                    self._channel.notify(
                        event="message_received",
                        by=client,
                        message=Message(
                            payload=grads[i].detach(),
                            msg_type="vfl_grad",
                            sender=self.server,
                        ),
                    )
                    client.backward_step(grads[i])

            # Step schedulers after each epoch
            for client in self.clients:
                client.scheduler.step()
            self.server.scheduler.step()

        return running_loss / max(n_batches, 1)

    def __str__(self) -> str:
        return (
            f"VerticalFL[{self._id}]("
            f"\n\tn_clients={self.n_clients},"
            f"\n\ttop_model={self.server.top_model.__class__.__name__},"
            f"\n\tbottom_model={self.clients[0].bottom_model.__class__.__name__},"
            f"\n\tfeatures_per_client={[len(fs) for fs in self.feature_splits]}"
            f"\n)"
        )

    def __repr__(self) -> str:
        return self.__str__()
