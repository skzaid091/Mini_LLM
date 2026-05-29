import torch
import torch.nn as nn
import math

class FeedForward(nn.Module):

    def __init__(self, embedding_dim, num_layers, hidden_dim=None, gelu_const=None, dropout_rate=0.1):

        super().__init__()

        self.embedding_dim = embedding_dim

        if hidden_dim is None:
            hidden_dim = embedding_dim * 4

        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(p=dropout_rate)

        self.GELU_CONST = gelu_const

        # =============================================
        # Linear Layers
        # =============================================

        self.W1 = nn.Parameter(
            torch.randn(
                embedding_dim,
                hidden_dim
            ) * 0.02
        )

        self.b1 = nn.Parameter(
            torch.zeros(hidden_dim)
        )

        self.W2 = nn.Parameter(
            torch.randn(
                hidden_dim,
                embedding_dim
            ) * (0.02 / math.sqrt(2 * num_layers))  # ← scaled down
        )

        self.b2 = nn.Parameter(
            torch.zeros(embedding_dim)
        )

    def gelu(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                self.GELU_CONST * (
                    x + 0.044715 * torch.pow(x, 3)
                )
            )
        )

    def forward(self, x):

        x = x @ self.W1 + self.b1

        x = self.gelu(x)

        x = self.dropout(x)

        x = x @ self.W2 + self.b2

        return x