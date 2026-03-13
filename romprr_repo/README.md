# Federated Learning Labs (romprr_repo)

This folder is the working repository for the labs and experiment tracking.

It contains:
- experiment configurations,
- custom datasets and custom tabular/VFL models,
- run outputs (CSV logs),
- analysis notebooks for each lab.

> Scope note: this README documents `romprr_repo` only. The vendored `fluke_package` is not documented here, except for the vertical FL extension used in this repo.

---

## 1) Repository structure

```text
romprr_repo/
├── config/                     # Experiment and algorithm YAML files
├── data/                       # Raw and cached datasets
├── dataset/                    # Custom dataset loaders
├── model/                      # Custom tabular + vertical models
├── runs/                       # CSV logs + plotting notebooks
├── VFL_IMPLEMENTATION_GUIDE.md # Detailed VFL implementation notes
├── test_import.py              # VerticalFL import smoke test
└── test_config.py              # Python API experiment script (fairness eval prototype)
```

---

## 2) Setup

From `romprr_repo/`:

1. Create/activate a Python environment.
2. Install dependencies from `fluke_package/requirements.txt`.
3. Install Fluke in editable mode from `fluke_package/`.

If installation is correct, CLI entrypoint should be available via:
- `python -m fluke.run`

---

## 3) Datasets used in labs

### Lab 1
- `mnist` (built-in dataset name in config)
- `adult` (built-in dataset name in config)

### Lab 2
- `adult` (built-in dataset name in config)

### Lab 3 and VFL
- Based on a Kaggle dataset (already in data/) : [available here](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- `dataset.Med_data.CARDIO_DATASET`
	- expected raw file: `data/cardio_train.csv`
	- cached intermediate files: `data/X_cardio.csv`, `data/y_cardio.csv`

---

## 4) How experiments are launched

CLI modes used in this repo:
- `federation` for standard horizontal FL (FedAvg/SCAFFOLD/DPFedAvg)
- `decentralized` for decentralized FedAvg
- `vertical` for VFL split learning

Typical usage pattern:

```bash
python -m fluke.run <mode> config/<exp_cfg>.yaml config/<alg_cfg>.yaml
```

Examples:

```bash
# Lab 1 (horizontal FL)
python -m fluke.run federation config/lab_1_iid_exp.yaml config/lab_1_fedavg_alg.yaml

# Lab 1 (decentralized)
python -m fluke.run decentralized config/lab_1_decentralized_fedavg_exp.yaml config/lab_1_decentralized_fedavg_alg.yaml

# VFL
python -m fluke.run vertical config/lab_vfl_exp.yaml config/lab_vfl_alg.yaml
```

All run outputs are written under `runs/<experiment_dir>/` from the logger `CsvLog`.

---

## 5) Lab-by-lab experiment index

## Lab 1 — MNIST + Adult (FedAvg, SCAFFOLD, Decentralized FedAvg)

Notebook:
- [runs/lab_1_plots.ipynb](runs/lab_1_plots.ipynb)

Main configurations:
- FedAvg (MNIST):
	- `config/lab_1_iid_exp.yaml` + `config/lab_1_fedavg_alg.yaml`
	- `config/lab_1_non_iid_exp.yaml` + `config/lab_1_fedavg_alg.yaml`
- SCAFFOLD (MNIST non-IID):
	- `config/lab_1_scaffold_exp.yaml` + `config/lab_1_scaffold_alg.yaml`
- Decentralized FedAvg (MNIST non-IID):
	- `config/lab_1_decentralized_fedavg_exp.yaml` + `config/lab_1_decentralized_fedavg_alg.yaml`
- Adult tabular baseline:
	- `config/lab_1_adult_exp.yaml` + `config/lab_1_adult_alg.yaml`

Related run folders (examples):
- `runs/lab_1_fluke_mnist_iid_traces-lep-30/`
- `runs/lab_1_fluke_mnist_noniid_traces-lep-30/`
- `runs/lab_1_decentralized_fedavg/`
- `runs/lab_1_scaffold/`
- `runs/lab_1_adult/`
- plus parameter sweeps in `runs/client_ratio/` and `runs/local_epochs/`

---

## Lab 2 — Custom tabular dataset/model

Notebook:
- [runs/lab_2_plots.ipynb](runs/lab_2_plots.ipynb)

Main configurations:
- `config/lab_2_tabular_exp.yaml` + `config/lab_2_tabular_alg.yaml`

Custom code used:
- dataset loader: `dataset/tabular_dataset.py`
- model definitions: `model/tabular_models.py`

Related run folders (examples):
- `runs/lab_2_tabular_LogReg_experiment_10_epoch/`
- `runs/lab_2_tabular_MLP_experiment_10_epoch/`
- `runs/local_epochs` -> contains all experiments with local epochs parameter
- `runs/client_ratio` -> contains all experiments with clients ratio parameter

---

## Lab 3 — Medical tabular FL (CARDIO), model and DP studies

Notebook:
- [runs/lab_3_plots.ipynb](runs/lab_3_plots.ipynb)

Experiment families:

1. **Model comparison (non-private):**
	 - alg configs:
		 - `config/lab_3_tabular_med_logreg.yaml`
		 - `config/lab_3_tabular_med_mlp.yaml`
		 - `config/lab_3_tabular_med_svm.yaml`
	 - exp configs:
		 - `config/lab_3_tabular_med_exp_iid_logreg.yaml`
		 - `config/lab_3_tabular_med_exp_iid.yaml` (MLP)
		 - `config/lab_3_tabular_med_exp_iid_svm.yaml`
		 - `config/lab_3_tabular_med_exp_non_iid.yaml`

2. **Differential privacy (DPFedAvg):**
	 - alg configs:
		 - `config/lab_3_tabular_med_DP_logreg.yaml`
		 - `config/lab_3_tabular_med_DP_mlp.yaml`
		 - `config/lab_3_tabular_med_DP_svm.yaml`
	 - exp base:
		 - `config/lab_3_tabular_med_exp_iid_DP.yaml`
	 - run folders indicate noise multiplier sweeps, e.g.:
		 - `runs/lab_3_med_experiment_iid_cardio_DP_mlp_1/`
		 - `runs/lab_3_med_experiment_iid_cardio_DP_mlp_5/`
		 - `runs/lab_3_med_experiment_iid_cardio_DP_mlp_10/`
		 - `runs/lab_3_med_experiment_iid_cardio_DP_mlp_100/`
		 - analogous folders for SVM.
 - **Fairness assessment:**
   - For this use case, we used this file for the MLP IID
   - python : `fairness_metric_run_iid_mlp.py`
     - File contains all the code to run an experiment with a custom evaluator

Main dataset/model code:
- `dataset/Med_data.py`
- `model/tabular_models.py`

---

## 6) Vertical FL (added part)

VFL configs:
- `config/lab_vfl_exp.yaml`
- `config/lab_vfl_alg.yaml`

VFL model code:
- `model/vertical_models.py`

Detailed implementation guide:
- [VFL_IMPLEMENTATION_GUIDE.md](VFL_IMPLEMENTATION_GUIDE.md)

Run output folder:
- `runs/lab_vfl_cardio/`

### VFL training flow summary

For each aligned mini-batch:
1. each client computes a local embedding from its feature slice;
2. server concatenates embeddings, computes logits and loss;
3. server backpropagates and sends embedding gradients back;
4. clients backpropagate through bottom models and update locally.

Only embeddings and embedding gradients are exchanged (no raw features or labels).

---