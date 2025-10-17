import torch
import torch.nn.functional as F
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q: (..., Lq, d)
    k: (..., Lk, d)
    v: (..., Lk, dv)
    mask: broadcastable a (..., Lq, Lk)
          - bool: True = BLOQUEAR (poner -inf)
          - float: 1.0 = permitir, 0.0 = bloquear
    Returns:
        output: (..., Lq, dv)
        attn:   (..., Lq, Lk)

    """
    # (..., Lq, Lk)
    scores = torch.matmul(q, k.transpose(-2, -1))
    dk = q.size(-1)
    scores = scores / dk**0.5

    if mask is not None:
        # Normalizamos a un tensor float con -inf donde se bloquea
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask, float("-inf"))
        else:
            # asumimos máscara en {0,1}: 0 = bloquear
            scores = scores.masked_fill(mask <= 0, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser múltiplo de num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # (B, L, d_model) -> (B, num_heads, L, d_head)
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_head).transpose(1, 2)

    def _combine_heads(self, x):
        # (B, num_heads, L, d_head) -> (B, L, d_model)
        B, H, L, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * D)

    def forward(self, x_q, x_kv, mask=None):
        #Primera proyeccion
        q = self._split_heads(self.w_q(x_q))
        k = self._split_heads(self.w_k(x_kv))
        v = self._split_heads(self.w_v(x_kv))


        if mask is not None:
        # Aceptamos:
        # (B, Lk), (B, Lq, Lk), (B, 1, Lq, Lk), (B, H, Lq, Lk)
          if mask.dim() == 2:
              mask = mask[:, None, None, :]

          elif mask.dim() == 3:
              mask = mask[:, None, :, :]

          elif mask.dim() == 4:
              pass # Ya funciona asi
          else:
              raise ValueError(f"Máscara con dims no soportadas: {mask.shape}")

        # Si float {0,1}
        if mask.dtype != torch.bool:
            mask = (mask <= 0)

        # Aplicamos Atencion y concatenamos
        attn_out, _ = scaled_dot_product_attention(q, k, v, mask)
        attn_out = self._combine_heads(attn_out)

        # Proyeccion Final
        attn_out = self.w_o(attn_out)
        attn_out = self.dropout(attn_out)
        return attn_out