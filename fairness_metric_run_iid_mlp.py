from fluke_package.fluke.data import DataSplitter
from fluke_package.fluke import DDict, FlukeENV
from fluke_package.fluke.utils.log import CsvLog
from fluke_package.fluke.evaluation import ClassificationEval
from fluke_package.fluke import FlukeENV
from fluke_package.fluke.algorithms.fedavg import FedAVG
import os
import time
import numpy as np
import torch

from dataset.Med_data import CARDIO_DATASET
from model.tabular_models import Custom_MLP


# Parameters that i usually change in my config
EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 256
WEIGHT_DECAY = 1e-4
N_CLIENTS = 10
FL_ROUNDS = 50
ELIGIBLE_PERC = 1.0
N_CLASSES=2
LOG_DIR="./runs/lab_3_med_experiment_SPD_EOD_iid_mlp"

# Sensitive feature indices for CARDIO after dropping `id`:
# [age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]
AGE_COL_IDX = 0
GENDER_COL_IDX = 1

# Fairness settings
POSITIVE_LABEL = 1
AGE_THRESHOLD_DAYS = 50 * 365  # age is provided in days
GENDER_PRIVILEGED_VALUE = 2.0  # in CARDIO: 1=female, 2=male


class FairClassificationEval(ClassificationEval):
  """Classification evaluator extended with fairness metrics for sensitive attributes.

  Added metrics:
  - `spd_age`, `eod_age`
  - `spd_gender`, `eod_gender`
  """

  def __init__(
    self,
    eval_every: int,
    n_classes: int,
    age_col_idx: int,
    gender_col_idx: int,
    positive_label: int = 1,
    age_threshold_days: int = 50 * 365,
    gender_privileged_value: float = 2.0,
    **metrics,
  ):
    super().__init__(eval_every=eval_every, n_classes=n_classes, **metrics)
    self.age_col_idx = age_col_idx
    self.gender_col_idx = gender_col_idx
    self.positive_label = int(positive_label)
    self.age_threshold_days = float(age_threshold_days)
    self.gender_privileged_value = float(gender_privileged_value)

  @staticmethod
  def _safe_rate(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    if denominator.item() == 0:
      return 0.0
    return (numerator / denominator).item()

  def _pred_to_labels(self, y_hat: torch.Tensor) -> torch.Tensor:
    if y_hat.ndim == 1:
      return (y_hat >= 0.5).long()
    if y_hat.shape[-1] == 1:
      return (y_hat.squeeze(-1) >= 0.5).long()
    return torch.argmax(y_hat, dim=1)

  def _compute_group_fairness_metrics(
    self,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    privileged_mask: torch.Tensor,
    unprivileged_mask: torch.Tensor,
  ) -> dict[str, float]:
    pred_pos = (y_pred == self.positive_label)
    true_pos = (y_true == self.positive_label)

    p_pred_pos_unpriv = self._safe_rate(
      (pred_pos & unprivileged_mask).sum(),
      unprivileged_mask.sum(),
    )
    p_pred_pos_priv = self._safe_rate(
      (pred_pos & privileged_mask).sum(),
      privileged_mask.sum(),
    )
    spd = p_pred_pos_unpriv - p_pred_pos_priv

    unpriv_tp_mask = unprivileged_mask & true_pos
    priv_tp_mask = privileged_mask & true_pos

    tpr_unpriv = self._safe_rate(
      (pred_pos & unpriv_tp_mask).sum(),
      unpriv_tp_mask.sum(),
    )
    tpr_priv = self._safe_rate(
      (pred_pos & priv_tp_mask).sum(),
      priv_tp_mask.sum(),
    )
    eod = tpr_unpriv - tpr_priv

    return {
      "p_pred_pos_unpriv": float(np.round(p_pred_pos_unpriv, 5)),
      "p_pred_pos_priv": float(np.round(p_pred_pos_priv, 5)),
      "tpr_unpriv": float(np.round(tpr_unpriv, 5)),
      "tpr_priv": float(np.round(tpr_priv, 5)),
      "spd": float(np.round(spd, 5)),
      "eod": float(np.round(eod, 5)),
    }

  @torch.no_grad()
  def evaluate(
    self,
    round: int,
    model: torch.nn.Module,
    eval_data_loader,
    loss_fn: torch.nn.Module | None = None,
    additional_metrics: dict | None = None,
    device: torch.device = torch.device("cpu"),
  ) -> dict:
    result = super().evaluate(
      round=round,
      model=model,
      eval_data_loader=eval_data_loader,
      loss_fn=loss_fn,
      additional_metrics=additional_metrics,
      device=device,
    )

    # Respect parent evaluator schedule/guards.
    if not result:
      return result

    model_device = torch.device("cpu")
    if next(model.parameters(), None) is not None:
      model_device = next(model.parameters()).device

    model.eval()
    model.to(device)

    all_X, all_y_true, all_y_pred = [], [], []

    if not isinstance(eval_data_loader, list):
      eval_data_loader = [eval_data_loader]

    for data_loader in eval_data_loader:
      for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        all_X.append(X.detach().cpu())
        all_y_true.append(y.detach().cpu())
        all_y_pred.append(self._pred_to_labels(y_hat.detach().cpu()))

    model.to(model_device)

    X_all = torch.cat(all_X, dim=0)
    y_true_all = torch.cat(all_y_true, dim=0).squeeze()
    y_pred_all = torch.cat(all_y_pred, dim=0).squeeze()

    age = X_all[:, self.age_col_idx]
    gender = X_all[:, self.gender_col_idx]

    # Age privileged group: younger than threshold (age is in days)
    age_privileged = age < self.age_threshold_days
    age_unprivileged = ~age_privileged

    # Gender privileged group: configured value (for CARDIO usually 2.0)
    gender_privileged = gender == self.gender_privileged_value
    gender_unprivileged = ~gender_privileged

    age_metrics = self._compute_group_fairness_metrics(
      y_true_all,
      y_pred_all,
      privileged_mask=age_privileged,
      unprivileged_mask=age_unprivileged,
    )
    gender_metrics = self._compute_group_fairness_metrics(
      y_true_all,
      y_pred_all,
      privileged_mask=gender_privileged,
      unprivileged_mask=gender_unprivileged,
    )

    result.update(
      {
        # Gap metrics
        "spd_age": age_metrics["spd"],
        "eod_age": age_metrics["eod"],
        "spd_gender": gender_metrics["spd"],
        "eod_gender": gender_metrics["eod"],
        # Group-specific rates (useful to interpret the gaps)
        "p_pred_pos_age_priv": age_metrics["p_pred_pos_priv"],
        "p_pred_pos_age_unpriv": age_metrics["p_pred_pos_unpriv"],
        "tpr_age_priv": age_metrics["tpr_priv"],
        "tpr_age_unpriv": age_metrics["tpr_unpriv"],
        "p_pred_pos_gender_priv": gender_metrics["p_pred_pos_priv"],
        "p_pred_pos_gender_unpriv": gender_metrics["p_pred_pos_unpriv"],
        "tpr_gender_priv": gender_metrics["tpr_priv"],
        "tpr_gender_unpriv": gender_metrics["tpr_unpriv"],
      }
    )

    return result

dataset = CARDIO_DATASET(path = "data")

env = FlukeENV()
env.set_seed(42) # we set a seed for reproducibility
env.set_device("cpu")
env.set_eval_cfg(
  pre_fit=True,
  post_fit=True,
  locals=True,
  server=True,
)

evaluator = FairClassificationEval(
    eval_every=1,
  n_classes=N_CLASSES,
  age_col_idx=AGE_COL_IDX,
  gender_col_idx=GENDER_COL_IDX,
  positive_label=POSITIVE_LABEL,
  age_threshold_days=AGE_THRESHOLD_DAYS,
  gender_privileged_value=GENDER_PRIVILEGED_VALUE,
)
# TODO : Add other metrics later

env.set_evaluator(evaluator)

splitter = DataSplitter(
    dataset,
  distribution="iid",
  client_split=0.5,
  sampling_perc=1.0,
  server_test=True,
  keep_test=True,
  server_split=0.0,
  uniform_test=False,
)

client_hp = DDict(
    batch_size=BATCH_SIZE,
    local_epochs=EPOCHS,
  persistency=True,
    loss="CrossEntropyLoss",
    optimizer=DDict(
      name="AdamW",
      lr=LR,
      # momentum=0.9,
      weight_decay=WEIGHT_DECAY),
    scheduler=DDict(
      gamma=1,
      step_size=1)
)

hyperparams = DDict(client=client_hp,
                    server=DDict(weighted=True),
                    model=Custom_MLP(input_dim=11, output_size=N_CLASSES))

algorithm = FedAVG(n_clients=N_CLIENTS,
                   data_splitter=splitter,
                   hyper_params=hyperparams)

logger = CsvLog(log_dir=LOG_DIR)
algorithm.set_callbacks(logger)

print(f"Logs will be saved to: {os.path.abspath(LOG_DIR)}")
start_time = time.perf_counter()
try:
  algorithm.run(n_rounds=FL_ROUNDS, eligible_perc=ELIGIBLE_PERC)
finally:
  logger.add_scalar("run_time_seconds", time.perf_counter() - start_time, 0)
  # Needed when using the Python API directly (without fluke.run wrappers).
  logger.close()



