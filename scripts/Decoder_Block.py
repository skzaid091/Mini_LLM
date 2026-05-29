import torch
import torch.nn as nn

from .Multi_Head_Causal_Attention import MultiHeadCausalSelfAttention
from .Feed_Forward import FeedForward
from .Layer_Normalization import LayerNorm

class DecoderBlock(nn.Module):
    
    def __init__(self, embedding_dim, max_seq_len, num_att_heads, gelu_const, dropout_rate, num_layers):

        super().__init__()
        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim,
            max_seq_len, 
            num_att_heads,
            dropout_rate
        )

        self.ffnn = FeedForward(
            embedding_dim=embedding_dim, 
            hidden_dim=embedding_dim*4, 
            gelu_const=gelu_const,
            dropout_rate=dropout_rate,
            num_layers=num_layers
        )

        self.layer_norm_1 = LayerNorm(embedding_dim)
        self.layer_norm_2 = LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, x, attention_mask):
        # Pre-LN Attention + Residual
        normed_x = self.layer_norm_1(x)

        attention_out = self.attention(normed_x, attention_mask)
        attention_out = self.dropout(attention_out)
        x = x + attention_out
    
        # Pre-LN FFNN + Residual
        normed_x = self.layer_norm_2(x)
        
        ffn_out = self.ffnn(normed_x)
        x = x + ffn_out
    
        return x