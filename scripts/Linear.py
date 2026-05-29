import torch
import torch.nn as nn

class Linear(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # ==========================================
        # Weight Initialization
        # ==========================================
        self.weights = nn.Parameter(
            torch.randn(
                in_features,
                out_features
            ) * (1 / (in_features ** 0.5))
        )

        # ==========================================
        # Bias Initialization
        # ==========================================
        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )


    def forward(self, x):

        """
        x shape:
        (batch_size, seq_len, in_features)
        """

        out = torch.matmul(
            x, 
            self.weights
        )
        out = out + self.bias

        return out