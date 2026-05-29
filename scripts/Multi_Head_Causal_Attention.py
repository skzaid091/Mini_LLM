import math
import torch
import torch.nn as nn

class MultiHeadCausalSelfAttention(nn.Module):

    def __init__(self, embedding_dim, max_seq_len, num_heads, dropout_rate):

        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dropout = nn.Dropout(p=dropout_rate)

        # ============================================
        # Projection Matrices
        # ============================================

        std = 1.0 / math.sqrt(embedding_dim)

        self.W_q = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * std)
        self.W_k = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * std)
        self.W_v = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * std)
        self.W_o = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * std)

        self.W_ob = nn.Parameter(
            torch.zeros(
                embedding_dim
            )
        )

        # ============================================
        # Causal Mask
        # ============================================

        self.register_buffer(
            'mask', 
            torch.tril(
                torch.ones(max_seq_len, max_seq_len)
            )
        )

    def forward(self, x, attention_mask):

        # x shape:
        # (B,T,C)

        B, T, C = x.shape

        # ============================================
        # Create Q,K,V
        # ============================================

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # ============================================
        # Split into heads
        # ============================================

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Shape:
        # (B,H,T,D)

        # ============================================
        # Attention Scores
        # ============================================

        raw_scores = Q @ K.transpose(-2, -1)
        scores = raw_scores / (self.head_dim ** 0.5)
        
        # Shape:
        # (B,H,T,T)

        # ============================================
        # Apply Causal Mask
        # ============================================

        mask = self.mask[:T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # ============================================
        # Padding Mask
        # ============================================

        # (B,T) -> (B,1,1,T)
        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask == 0, float("-inf"))

        # ============================================
        # Softmax
        # ============================================

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # ============================================
        # Weighted Sum
        # ============================================

        output = attention_weights @ V

        # ============================================
        # Merge Heads
        # ============================================

        output = output.transpose(1, 2).contiguous().view(B, T, C)

        # ============================================
        # Output Projection
        # ============================================

        output = (output @ self.W_o) + self.W_ob

        return output