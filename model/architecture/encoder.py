import torch
import torch.nn.functional as F
import torch.nn as nn
from model.architecture.attention import *
import math
from model.architecture.pos_encoding import *


class PositionwiseFeedForward(nn.Module):
    """
    Feed-Forward Network del paper Attention Is All You Need.
    Implementa: FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.activation(x1)
        x3 = self.dropout(x2)
        x4 = self.fc2(x3)
        return x4

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, layernorm_eps: float = 1e-6):

        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layernorm_eps) # aplica una normalización por capa a lo largo de las dimensiones de características
        self.norm2 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        #  Multi-head self-attention
        attn_out = self.mha(x_q=x, x_kv=x, mask=mask)
        x = self.norm1(x + attn_out)

        #  Feed-forward
        ffn_out = self.ffn(x)
        ffn_out = self.dropout_ffn(ffn_out)
        x = self.norm2(x + ffn_out)
        return x
    

class Encoder(nn.Module):
    """
    Embedding -> escala sqrt(d_model) -> + PositionalEncoding( fijo ) -> Dropout
    -> N x EncoderLayer (post-norm como en Vaswani)
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 vocab_size: int, max_pos: int, dropout: float = 0.1, layernorm_eps: float = 1e-6):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # Positional encoding
        pe = positional_encoding(max_pos, d_model)
        self.register_buffer("pe", pe, persistent=False)   # no entrenable

        # Pila de capas encoders
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         dropout=dropout,
                         layernorm_eps=layernorm_eps)
            for _ in range(num_layers)])

    def forward(self, x_ids, src_key_padding_mask: torch.Tensor | None = None):
        """
        x_ids: (B, L) ids de tokens
        src_key_padding_mask: (B, L) bool, True = padding => bloquear
        """
        # Embeddings Con posicion por POS encoding
        x = self.embedding(x_ids) * math.sqrt(self.d_model)
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        x = self.dropout(x)

        # Capas de atencion
        for layer in self.layers:
            x = layer(x, mask=src_key_padding_mask)

        return x  # (B, L, d_model)