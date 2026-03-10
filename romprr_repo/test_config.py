from fluke_package.fluke.data import DataSplitter
from fluke_package.fluke import DDict, FlukeENV
from fluke_package.fluke.utils.log import CsvLog
from fluke_package.fluke.evaluation import ClassificationEval
from fluke_package.fluke import FlukeENV
from fluke_package.fluke.algorithms.fedavg import FedAVG

from dataset.Med_data import CDC_DIABETES_INDICATORS
from model.tabular_models import Custom_MLP


# Parameters that i usually change in my config
EPOCHS = 5
LR = 1e-4
BATCH_SIZE = 10
WEIGHT_DECAY = 1e-4
N_CLIENTS = 10
FL_ROUNDS = 50
ELIGIBLE_PERC = 1.0
N_CLASSES=3
LOG_DIR="./runs/python_api_medical_test"

dataset = CDC_DIABETES_INDICATORS(path = "data")

env = FlukeENV()
env.set_seed(42) # we set a seed for reproducibility
env.set_device("cpu")

evaluator = ClassificationEval(
    eval_every=1,
    n_classes=N_CLASSES
)
# TODO : Add other metrics later

env.set_evaluator(evaluator)

splitter = DataSplitter(
    dataset,
    distribution="iid"
)

client_hp = DDict(
    batch_size=BATCH_SIZE,
    local_epochs=EPOCHS,
    loss="CrossEntropyLoss",
    optimizer=DDict(
      lr=LR,
      momentum=0.9,
      weight_decay=WEIGHT_DECAY),
    scheduler=DDict(
      gamma=1,
      step_size=1)
)

hyperparams = DDict(client=client_hp,
                    server=DDict(weighted=True),
                    model=Custom_MLP(input_dim=21, output_size=3))

algorithm = FedAVG(n_clients=N_CLIENTS,
                   data_splitter=splitter,
                   hyper_params=hyperparams)

logger = CsvLog(log_dir=LOG_DIR)
algorithm.set_callbacks(logger)

algorithm.run(n_rounds=FL_ROUNDS, eligible_perc=ELIGIBLE_PERC)



