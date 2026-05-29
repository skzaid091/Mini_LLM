import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len, embedding_dim):

        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =================================================
        # Create PE Matrix
        # =================================================

        pe = torch.zeros(
            max_seq_len,
            embedding_dim,
            device=self.device
        )

        # Positions
        position = torch.arange(
            0,
            max_seq_len,
            dtype=torch.float
        ).unsqueeze(1).to(self.device)

        # Frequency scaling
        div_term = torch.exp(
            torch.arange(
                0,
                embedding_dim,
                2
            ).float()
            * (
                -torch.log(
                    torch.tensor(10000.0)
                ) / embedding_dim
            )
        ).to(self.device)

        # Even indices -> sin
        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        # Odd indices -> cos
        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # Add batch dimension
        self.register_buffer('pe', pe.unsqueeze(0))
        

    def forward(self, x):

        # x shape:
        # (batch_size, seq_len, embedding_dim)

        seq_len = x.size(1)

        return x + self.pe[:, :seq_len]