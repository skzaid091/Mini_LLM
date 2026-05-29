import torch
import torch.nn as nn

class Embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =================================================
        # Embedding Matrix
        # =================================================

        self.weight = nn.Parameter(
            torch.randn(
                vocab_size,
                embedding_dim,
                device=self.device
            ) * 0.02
        )

    def forward(self, input_ids):

        # input_ids:
        # (batch_size, seq_len)

        return self.weight[input_ids]