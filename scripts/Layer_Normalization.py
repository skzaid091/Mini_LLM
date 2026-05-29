import torch
import torch.nn as nn

class LayerNorm(nn.Module):

    def __init__(self, embedding_dim, eps=1e-5):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.eps = eps

        # =============================================
        # Learnable Parameters
        # =============================================

        self.gamma = nn.Parameter(
            torch.ones(embedding_dim)
        )

        self.beta = nn.Parameter(
            torch.zeros(embedding_dim)
        )

    def forward(self, x):

        # x:
        # (B,T,C)

        # =============================================
        # Mean
        # =============================================

        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        # =============================================
        # Variance
        # =============================================

        variance = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False
        )

        # =============================================
        # Normalize
        # =============================================

        x_hat = (
            x - mean
        ) / torch.sqrt(
            variance + self.eps
        )

        # =============================================
        # Scale + Shift
        # =============================================

        out = (
            self.gamma * x_hat
        ) + self.beta

        return out