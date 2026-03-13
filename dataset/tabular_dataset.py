import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from fluke.data.datasets import DataContainer

def SimpleTabularDataset(**kwargs) -> DataContainer:
    # 1. Pull the raw tabular dataset directly from sklearn
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # 2. Split it (Absolutely no transformations added)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Convert straight to Tensors and return Fluke's container
    return DataContainer(
        X_train=torch.tensor(X_train, dtype=torch.float32),
        y_train=torch.tensor(y_train, dtype=torch.long),
        X_test=torch.tensor(X_test, dtype=torch.float32),
        y_test=torch.tensor(y_test, dtype=torch.long),
        num_classes=2
    )