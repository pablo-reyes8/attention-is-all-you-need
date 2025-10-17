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


class DecoderLayer(nn.Module):
    """
    Bloque del decoder:
      1) Masked self-attention (look-ahead)
      2) Cross-attention (Q=salida del 1, K/V=encoder_out)
      3) FFN
    Con Add & Norm (post-norm), como en Vaswani y tu TF.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, layernorm_eps: float = 1e-6):
        super().__init__()

        self.mha1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.mha2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn  = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layernorm_eps)

        self.drop_ffn = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                enc_output: torch.Tensor,
                look_ahead_mask: torch.Tensor | None = None,
                enc_padding_mask: torch.Tensor | None = None,
                return_attn: bool = False):
        """
        x: (B, L_tgt, d_model)
        enc_output: (B, L_src, d_model)
        look_ahead_mask: bool/float, broadcastable a (B, h, L_tgt, L_tgt). True/0 = bloquear
        enc_padding_mask: bool/float, broadcastable a (B, h, L_tgt, L_src). True/0 = bloquear
        return_attn: si True e implementado en MHA, retorna mapas de atención
        """

        # Bloque 1 masked self-attention
        if return_attn and "return_attn" in self.mha1.forward.__code__.co_varnames:
            out1, attn_w1 = self.mha1(x_q=x, x_kv=x, mask=look_ahead_mask, return_attn=True)
        else:
            out1 = self.mha1(x_q=x, x_kv=x, mask=look_ahead_mask)
            attn_w1 = None
        x1 = self.norm1(x + out1)

        #  Bloque 2 cross-attention (Q=x1, K/V=enc_output)
        if return_attn and "return_attn" in self.mha2.forward.__code__.co_varnames:
            out2, attn_w2 = self.mha2(x_q=x1, x_kv=enc_output, mask=enc_padding_mask, return_attn=True)
        else:
            out2 = self.mha2(x_q=x1, x_kv=enc_output, mask=enc_padding_mask)
            attn_w2 = None
        x2 = self.norm2(x1 + out2)

        #  Bloque 3: FFN
        ffn_out = self.ffn(x2)
        ffn_out = self.drop_ffn(ffn_out)
        y = self.norm3(x2 + ffn_out)

        return y, attn_w1, attn_w2
    

class Decoder(nn.Module):
    """
    Embedding -> escala sqrt(d_model) -> + PosEnc (fijo) -> Dropout
    -> N x DecoderLayer (masked self-attn, cross-attn, FFN)  [post-norm]
    """

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 target_vocab_size: int, max_pos: int,
                 dropout: float = 0.1, layernorm_eps: float = 1e-6):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # PE
        pe = positional_encoding(max_pos, d_model)
        self.register_buffer("pe", pe, persistent=False)

        # Repetir las decoder layers n veces
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         dropout=dropout,
                         layernorm_eps=layernorm_eps)
            for _ in range(num_layers)])

    def forward(self, x_ids: torch.Tensor,
                enc_output: torch.Tensor,
                look_ahead_mask: torch.Tensor | None = None,
                enc_padding_mask: torch.Tensor | None = None,
                return_attn: bool = True):
        """
        Returns:
          x: (B, L_tgt, d_model)
          attn_weights: dict[str, Tensor] con mapas de atención (o None si no se piden)
        """

        B, L_tgt = x_ids.size(0), x_ids.size(1)

        # Embedding con escala con PE
        x = self.embedding(x_ids) * math.sqrt(self.d_model)
        x = x + self.pe[:, :L_tgt, :] # suma PE
        x = self.dropout(x)

        attn_weights = {}
        for i, layer in enumerate(self.dec_layers, start=1):
            x, attn_w1, attn_w2 = layer(x=x,
                enc_output=enc_output,
                look_ahead_mask=look_ahead_mask,
                enc_padding_mask=enc_padding_mask,

                return_attn=return_attn)

            if return_attn:
                attn_weights[f"decoder_layer{i}_block1_self_att"] = attn_w1
                attn_weights[f"decoder_layer{i}_block2_decenc_att"] = attn_w2
            else:
              pass

        return x, attn_weights if return_attn else None