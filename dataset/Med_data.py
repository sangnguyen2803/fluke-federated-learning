import os
from sklearn.discriminant_analysis import StandardScaler
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from fluke_package.fluke.data.datasets import DataContainer
# from ucimlrepo import fetch_ucirepo 

# Need to update to just clean up the dataset (pull from kaggle & remove rows with label 1 and change label 2 to one after)
 
def CDC_DIABETES_INDICATORS(**kwargs) -> DataContainer :

    data_path = kwargs.get('path')

    os.makedirs(data_path, exist_ok=True)
    
    data_file = os.path.join(data_path, "cdc_dataset.csv")

    x_file = os.path.join(data_path, 'X.csv')
    y_file = os.path.join(data_path, 'y.csv')

    if os.path.exists(x_file) and os.path.exists(y_file):
        print(f"Loading dataset from local cache: {data_path}")
        X = pd.read_csv(x_file)
        y = pd.read_csv(y_file)

        print(f"Data set of {len(X)} columns.")

    else:
        print('Cleaning dataset from kaggle (please download it and paste it to ./data/cdc_dataset.csv)')
        
        df = pd.read_csv(data_file)

        df = df[df['Diabetes_012'] != 1]
        df['Diabetes_012'] = df['Diabetes_012'].replace(2, 1)

        y = pd.DataFrame(df.pop('Diabetes_012'))
        X = df

        print(f"Data set of {len(X)} columns.")

        X.to_csv(x_file, index=False)
        y.to_csv(y_file, index=False)
        print(f"Dataset saved to: {data_path}")

    # There are no categorical data and no missing data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scaler = StandardScaler()
    # # Fit on training data, transform both
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # There are 3 classes : (0) Healthy, (1) Prediabete, (2) Diabete
    return DataContainer(
        X_train=torch.tensor(X_train.to_numpy(), dtype=torch.float32),
        y_train=torch.tensor(y_train.to_numpy(), dtype=torch.long).squeeze(),
        X_test=torch.tensor(X_test.to_numpy(), dtype=torch.float32),
        y_test=torch.tensor(y_test.to_numpy(), dtype=torch.long).squeeze(),
        num_classes=2
    )

def CARDIO_DATASET(**kwargs) -> DataContainer :
    data_path = kwargs.get('path')

    os.makedirs(data_path, exist_ok=True)
    
    data_file = os.path.join(data_path, "cardio_train.csv")

    x_file = os.path.join(data_path, 'X_cardio.csv')
    y_file = os.path.join(data_path, 'y_cardio.csv')

    if os.path.exists(x_file) and os.path.exists(y_file):
        print(f"Loading dataset from local cache: {data_path}")
        X = pd.read_csv(x_file)
        y = pd.read_csv(y_file)

        print(f"Data set of {len(X)} columns.")

    else:
        print('Cleaning dataset from kaggle (please download it and paste it to ./data/cdc_dataset.csv)')
        
        df = pd.read_csv(data_file, sep=";")

        y = pd.DataFrame(df.pop('cardio'))
        X = df

        print(f"Data set of {len(X)} columns.")

        X.to_csv(x_file, index=False)
        y.to_csv(y_file, index=False)
        print(f"Dataset saved to: {data_path}")

    # There are no categorical data and no missing data
    X.pop('id')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scaler = StandardScaler()
    # # Fit on training data, transform both
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # There are 3 classes : (0) Healthy, (1) Prediabete, (2) Diabete
    return DataContainer(
        X_train=torch.tensor(X_train.to_numpy(), dtype=torch.float32),
        y_train=torch.tensor(y_train.to_numpy(), dtype=torch.long).squeeze(),
        X_test=torch.tensor(X_test.to_numpy(), dtype=torch.float32),
        y_test=torch.tensor(y_test.to_numpy(), dtype=torch.long).squeeze(),
        num_classes=2)

if __name__ == '__main__' :
    CDC_DIABETES_INDICATORS(path = "/home/romain/dev/Master/federated_learning/fl-lab/data")


