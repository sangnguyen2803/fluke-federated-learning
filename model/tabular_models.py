import torch
import torch.nn as nn
import torchvision.ops as ops
from torch.functional import F

# 1. The Standard MLP
class StandardMLP(nn.Module):
    def __init__(self, input_dim=30, num_classes=2):
        super().__init__()
        self.model = ops.MLP(in_channels=input_dim, hidden_channels=[64, 32, num_classes])

    def forward(self, x):
        return self.model(x)

# 2. TabNet (The Neural Decision Tree / XGBoost Alternative)
# from pytorch_tabnet.tab_network import TabNet
# class GoogleTabNet(nn.Module):
#     def __init__(self, input_dim=30, num_classes=2):
#         super().__init__()
#         self.model = TabNet(input_dim=input_dim, output_dim=num_classes)

#     def forward(self, x):
#         out, _ = self.model(x)
#         return out

# # 3. FT-Transformer (The Attention/Transformer approach)
# import rtdl
# class TabularTransformer(nn.Module):
#     def __init__(self, input_dim=30, num_classes=2):
#         super().__init__()
#         # We have to tell the transformer we have 30 continuous numerical features
#         self.model = rtdl.FTTransformer.make_baseline(
#             n_num_features=input_dim,
#             cat_cardinalities=None, # No categorical features in Breast Cancer dataset
#             d_out=num_classes,
#         )

#     def forward(self, x):
#         # rtdl expects separate x_num and x_cat inputs. We only have numeric (x_num).
#         return self.model(x_num=x, x_cat=None)

class Custom_SVM(nn.Module):
    """Linear SVM-style classifier for tabular datasets like Adult."""

    def __init__(self, input_dim: int | None = 30, output_size: int = 2):
        super().__init__()
        if input_dim is None:
            self.fc = nn.LazyLinear(output_size)
        else:
            self.fc = nn.Linear(input_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class Custom_LogReg(nn.Module):
    """Logistic regression for tabular datasets like Adult."""

    def __init__(self, input_dim: int | None = 30, output_size: int = 2):
        super().__init__()
        if input_dim is None:
            self.fc = nn.LazyLinear(output_size)
        else:
            self.fc = nn.Linear(input_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class Custom_MLP(nn.Module):
    """Simple MLP for tabular datasets like Adult."""

    def __init__(self, input_dim: int = 30, hidden1: int = 64, hidden2: int = 32, output_size: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
