"""Vertical data splitting utilities for Vertical Federated Learning (VFL).

In VFL, each client holds a different subset of **features** (columns) for the same set of
samples. Labels are typically held by one party — here, the server. This module provides
:class:`VerticalDataSplitter` which partitions the feature space across clients.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

import numpy as np
import torch

sys.path.append(".")
sys.path.append("..")

from . import DataContainer, FastDataLoader  # NOQA

__all__ = ["VerticalDataSplitter"]


class VerticalDataSplitter:
    """Splits a tabular dataset **vertically** (by features) across clients.

    Each client receives all samples but only a disjoint subset of features.
    The server retains the labels and (optionally) a test set for global evaluation.

    Args:
        dataset (DataContainer): The full dataset.
        client_split (float): Fraction of each client's data used for local test.
            Defaults to ``0.0`` (no local test split — clients don't have labels anyway).
        sampling_perc (float): Percentage of samples to use. Defaults to ``1.0``.
        server_test (bool): Whether the server keeps a global test set. Defaults to ``True``.
        feature_splits (list[list[int]] | None): Optional manual feature split.
            A list of lists, one per client, containing the column indices that client owns.
            E.g. ``[[0, 1, 2], [3, 4], [5, 6, 7]]``.  When ``None`` (default), features
            are randomly and evenly partitioned.
    """

    def __init__(
        self,
        dataset: DataContainer,
        server_test: bool = True,
        sampling_perc: float = 1.0,
        feature_splits: list[list[int]] | None = None,
        **kwargs,
    ):
        self.data_container = dataset
        self.server_test = server_test
        self.sampling_perc = sampling_perc
        self._manual_feature_splits = feature_splits

    @property
    def num_classes(self) -> int:
        return self.data_container.num_classes

    def assign(
        self, n_clients: int, batch_size: int = 32
    ) -> dict:
        """Partition features across *n_clients* clients.

        Returns a dict with keys:
            - ``"clients_train"``: list of ``FastDataLoader`` per client (X features only).
            - ``"clients_test"``: list of ``FastDataLoader`` per client (X features only).
            - ``"server_train"``: ``FastDataLoader`` with (labels only, aligned with client data).
            - ``"server_test"``: ``FastDataLoader`` with (X_full, y) for global eval (or ``None``).
            - ``"feature_splits"``: list of arrays, feature indices per client.
        """
        X_train, y_train = self.data_container.train
        X_test, y_test = self.data_container.test

        num_features = X_train.shape[1]
        assert n_clients <= num_features, (
            f"Cannot split {num_features} features among {n_clients} clients; "
            f"need n_clients <= num_features."
        )

        # ── Partition feature indices ──────────────────────────────────────
        if self._manual_feature_splits is not None:
            # Use the manually specified split
            feature_splits = [np.array(sorted(fs)) for fs in self._manual_feature_splits]
            assert len(feature_splits) == n_clients, (
                f"feature_splits has {len(feature_splits)} groups but n_clients={n_clients}"
            )
            all_indices = sorted(idx for fs in feature_splits for idx in fs)
            assert all_indices == list(range(num_features)), (
                f"feature_splits must cover all {num_features} features exactly once. "
                f"Got indices: {all_indices}"
            )
            print(f"FEATURES SPLIT (manual): {feature_splits}")
        else:
            # Random even partition
            perm = np.random.permutation(num_features)
            base_size = num_features // n_clients
            remainder = num_features % n_clients
            feature_splits = []
            offset = 0
            for i in range(n_clients):
                size = base_size + (1 if i < remainder else 0)
                feature_splits.append(np.sort(perm[offset: offset + size]))
                offset += size
            print(f"FEATURES SPLIT (auto): {feature_splits}")

        # ── Build per-client data loaders (features only, no labels) ──────
        clients_train = []
        clients_test = []
        for feat_idx in feature_splits:
            X_tr_client = X_train[:, feat_idx]
            clients_train.append(
                FastDataLoader(
                    X_tr_client,
                    num_labels=self.data_container.num_classes,
                    batch_size=batch_size,
                    shuffle=False,  # must stay aligned across clients
                    percentage=self.sampling_perc,
                )
            )
            X_te_client = X_test[:, feat_idx]
            clients_test.append(
                FastDataLoader(
                    X_te_client,
                    num_labels=self.data_container.num_classes,
                    batch_size=batch_size,
                    shuffle=False,
                    percentage=self.sampling_perc,
                )
            )

        # ── Server data loaders ───────────────────────────────────────────
        # Training labels (aligned with client data)
        server_train = FastDataLoader(
            y_train,
            num_labels=self.data_container.num_classes,
            batch_size=batch_size,
            shuffle=False,
        )

        # Full test set for global evaluation
        server_test = None
        if self.server_test:
            server_test = FastDataLoader(
                X_test,
                y_test,
                num_labels=self.data_container.num_classes,
                batch_size=128,
                shuffle=False,
            )

        return {
            "clients_train": clients_train,
            "clients_test": clients_test,
            "server_train": server_train,
            "server_test": server_test,
            "feature_splits": feature_splits,
        }
