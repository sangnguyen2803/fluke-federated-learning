"""Neural network architectures for Vertical Federated Learning (VFL).

In VFL each client trains a *bottom model* on its local feature subset and sends the
resulting embeddings to the server. The server owns a *top model* that takes the
concatenated embeddings from all clients and produces predictions.

The combined architecture mirrors :class:`model.tabular_models.Custom_MLP`:
    ``input → Linear(hidden1) → ReLU → Linear(hidden2) → ReLU``  (BottomModel)
    ``concat(embeddings) → Linear(output)``                       (TopModel)

So when combined end-to-end the network is:
    ``input → 64 → ReLU → 32 → ReLU → output``  (same as Custom_MLP)

Classes:
    BottomModel  – client-side: ``input → hidden1 → ReLU → hidden2 → ReLU``
    TopModel     – server-side: ``concat → output`` (single linear layer)
"""

import torch
import torch.nn as nn
from torch.functional import F


__all__ = ["BottomModel", "TopModel"]


class BottomModel(nn.Module):
    """Client-side bottom model — first two layers of Custom_MLP.

    Architecture: ``input_dim → hidden1 (ReLU) → hidden2 (ReLU)``

    Args:
        input_dim (int): Number of input features for this client.
        embedding_dim (int): Output embedding dimension (= hidden2 of Custom_MLP). Defaults to 32.
        hidden_dim (int): Width of the first hidden layer (= hidden1 of Custom_MLP). Defaults to 64.
    """

    def __init__(self, input_dim: int, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)      # hidden1
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)   # hidden2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # ReLU here too — matches Custom_MLP layer 2
        return x


class TopModel(nn.Module):
    """Server-side top model — final layer of Custom_MLP.

    Architecture: ``concat(embeddings) → output``  (single linear, no hidden layer)

    This keeps the combined BottomModel+TopModel equivalent to Custom_MLP:
        ``input → 64 → ReLU → 32 → ReLU → output``

    Args:
        input_dim (int): Total dimension of all concatenated embeddings
            (= n_clients × embedding_dim).
        output_dim (int): Number of output classes. Defaults to 2.
        hidden_dim (int): Unused — kept for interface compatibility.
    """

    def __init__(self, input_dim: int, output_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)  # single linear layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x)
