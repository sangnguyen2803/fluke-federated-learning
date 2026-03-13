# VFL in Our Fluke Framework — Implementation Guide

This document explains how Vertical Federated Learning (VFL) is implemented in this project, with concrete references to the code paths and practical notes.

---

## 1) Where the VFL implementation lives

- CLI command entrypoint (`vertical`): `fluke_package/fluke/run.py`
- Vertical run pipeline (`_run_vertical`): `fluke_package/fluke/run.py`
- Core algorithm and training logic: `fluke_package/fluke/algorithms/vertical.py`
- Vertical data partitioning: `fluke_package/fluke/data/vertical.py`
- Bottom/Top model definitions used in VFL experiments: `model/vertical_models.py`
- VFL experiment config: `config/lab_vfl_exp.yaml`
- VFL algorithm config: `config/lab_vfl_alg.yaml`

---

## 2) High-level design

Our VFL follows a split-learning style design:

- Each client owns a **subset of features** (columns), for the **same aligned samples**.
- Each client has a **bottom model** that maps local features to an embedding.
- The server has the labels and a **top model** that consumes concatenated embeddings.
- During training:
  1. clients send embeddings to server,
  2. server computes loss/backprop through top model,
  3. server sends gradient slices back,
  4. each client backprops through bottom model and updates.

No raw features or labels are exchanged across parties.

---

## 3) End-to-end run flow from CLI

### `fluke vertical ...`

The `vertical` command loads experiment and algorithm configs, composes overrides (if any), then calls `_run_vertical(cfg)`.

### `_run_vertical(cfg)`

Main steps:

1. Configure runtime env (`FlukeENV`).
2. Load dataset via `Datasets.get(**cfg.data.dataset)`.
3. Build evaluator (`ClassificationEval`).
4. Read optional manual feature splits from `cfg.data.feature_splits`.
5. Instantiate `VerticalDataSplitter`.
6. Instantiate algorithm class from `cfg.method.name` (here `fluke.algorithms.vertical.VerticalFL`).
7. Attach logger callbacks and run `fl_algo.run(cfg.protocol.n_rounds, cfg.protocol.eligible_perc)`.

---

## 4) Data handling and feature partitioning

`VerticalDataSplitter` in `fluke_package/fluke/data/vertical.py` is central.

### Key behavior

- Asserts `n_clients <= num_features`.
- Supports either:
  - manual feature groups (`feature_splits` in config), or
  - automatic random even partitioning.
- Validates manual split covers all feature indices exactly once.
- Creates one `FastDataLoader` per client for train/test **features only**.
- Creates server-side train loader with labels (`y_train`) and optional global test loader (`X_test, y_test`).

### Alignment guarantee

All VFL loaders are created with `shuffle=False` to preserve row-wise alignment across all clients and server labels.
This is critical in VFL because all parties must refer to the same sample order for each mini-batch.

---

## 5) Models used in this project

Defined in `model/vertical_models.py`.

### `BottomModel`

Client-side network:

- `Linear(input_dim -> hidden_dim)` + ReLU
- `Linear(hidden_dim -> embedding_dim)` + ReLU

### `TopModel`

Server-side network:

- `Linear(concat_dim -> output_dim)`

In our config:

- `embedding_dim = 32`
- `hidden_dim = 64`
- `output_dim = 2` (binary cardio classification)

Algorithm config points to:

- `model_bottom: model.vertical_models.BottomModel`
- `model_top: model.vertical_models.TopModel`
- `name: fluke.algorithms.vertical.VerticalFL`

---

## 6) Core VFL training logic (`VerticalFL`)

Implemented in `fluke_package/fluke/algorithms/vertical.py`.

### Construction phase (`VerticalFL.__init__`)

- Calls `data_splitter.assign(...)` to get:
  - `clients_train`, `clients_test`
  - `server_train` (labels)
  - `server_test` (global eval set)
  - `feature_splits`
- Builds one bottom model per client (input dim depends on that client’s feature slice).
- Builds one top model with input dim = `embedding_dim * n_clients`.
- Creates `VFLClient` objects and one `VFLServer`.
- Attaches a channel for communication events/logging.

### Round loop (`run`)

For each round:

1. Start round callback.
2. Train one round via `_train_one_round`.
3. If server evaluation enabled, run global eval.
4. End round callback.

---

## 7) Per-mini-batch mechanics (most important part)

Inside `_train_one_round`:

1. Iterate in sync using `zip(*client_iters, label_iter)`.
2. For each client:
   - forward local batch through bottom model,
   - cache embedding (`client._cached_embedding = emb`),
   - create detached leaf `emb_detached = emb.detach().requires_grad_(True)`,
   - send embedding message to server side channel events.
3. Server:
   - concatenates embeddings,
   - forward through top model,
   - compute loss with `y_batch`,
   - `loss.backward()`,
   - slice `concat.grad` into per-client gradient chunks,
   - update top model optimizer.
4. Each client receives its gradient slice:
   - `optimizer.zero_grad()`,
   - `_cached_embedding.backward(grad_embedding)`,
   - gradient clipping,
   - `optimizer.step()`.

This is split backpropagation: top-model gradients are converted into embedding-level gradients and propagated into each bottom model.

---

## 8) Why detach + cache is needed

Clients compute `emb` from their own graph. The server receives `emb_detached` as a new leaf tensor.

- `detach()` simulates communication boundary and decouples autograd graphs.
- `requires_grad_(True)` lets server compute gradients wrt embedding.
- Client retains original `_cached_embedding` (with full local graph), so receiving `dL/dEmb` is enough to backprop into local model weights.

This pattern is the key trick that enables VFL split learning in a single-process simulation while preserving role separation.

---

## 9) Server-side evaluation in VFL

`VFLServer.evaluate(...)` uses `_VFLInferenceModel` wrapper:

- Given full `X`, it selects each client’s feature subset (`feature_splits`),
- runs each bottom model,
- concatenates embeddings,
- applies top model,
- delegates metrics computation to Fluke evaluator.

So global metrics are end-to-end metrics of the composite VFL pipeline.

---

## 10) Communication accounting and what comm_cost means

Communication is tracked through channel observer callbacks:

- `Log.message_received(...)` adds `message.size` into `comm` tracker.

`Message.size` for tensor payloads is based on `numel()` (element count), not actual serialized bytes.
So `comm_costs.csv` is effectively in “number of float elements exchanged”, not strict network bytes.

In VFL this includes two message types per mini-batch:

- uplink embedding (`vfl_embedding`),
- downlink embedding gradient (`vfl_grad`).

Given fixed batch sizes and architecture, per-round comm cost tends to be constant (as seen in `runs/lab_vfl_cardio/comm_costs.csv`).

---

## 11) Your current experiment setup (cardio)

From config and dataset:

- Dataset: `dataset.Med_data.CARDIO_DATASET`
- `id` column is removed before train/test split.
- Number of classes: 2.
- Feature split in experiment config:
  - Client 0: `[0,1,2,3]`
  - Client 1: `[4,5,6,7]`
  - Client 2: `[8,9,10]`
- `n_clients = 3`
- `n_rounds = 50`
- `batch_size = 256`
- `local_epochs = 10`

Logs are written to `runs/lab_vfl_cardio`.

---

## 12) Subtle implementation details to remember for questions

1. **All clients participate every round** in current `VerticalFL` flow.
   - `eligible_perc` is accepted by `run(...)` but not used for client sampling logic.

2. **Alignment is mandatory**.
   - If any loader length differs, `zip(...)` truncates to shortest iterator.

3. **Local test on clients is not central in VFL**.
   - Server global eval is the meaningful metric path (`eval.server: true`, `eval.locals: false`).

4. **Communication metric unit** is tensor elements.
   - Useful for relative cost comparisons, not direct bandwidth billing.

5. **Gradient clipping** is enabled on both client bottom models and server top model (`max_norm=1.0`).

---

## 13) 30-second oral summary

“Our VFL implementation in Fluke is split learning with vertically partitioned tabular features. Each client trains a bottom network on its own columns and sends only embeddings to the server. The server owns labels and a top network, computes loss, backpropagates, then sends embedding gradients back so clients can update their local bottom models. Data alignment is enforced with non-shuffled loaders, evaluation is done end-to-end on server using an inference wrapper that reconstructs bottom+top forward passes, and communication cost is logged as tensor element counts exchanged per round.”

---

## 14) File map (quick lookup)

- `fluke_package/fluke/run.py` — command + run orchestration for VFL
- `fluke_package/fluke/algorithms/vertical.py` — clients/server/round logic
- `fluke_package/fluke/data/vertical.py` — vertical splitting and aligned loaders
- `model/vertical_models.py` — BottomModel and TopModel
- `config/lab_vfl_exp.yaml` — experiment setup (dataset, splits, rounds)
- `config/lab_vfl_alg.yaml` — VFL algorithm/model hyperparameters
- `runs/lab_vfl_cardio/*.csv` — produced metrics/communication/runtime
