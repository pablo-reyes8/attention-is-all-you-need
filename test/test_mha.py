import torch
import torch.nn as nn

from model.architecture.attention import MultiHeadAttention

def test_mha_shapes_and_grads():
    B, L, d_model, h = 3, 12, 128, 4
    x = torch.randn(B, L, d_model, requires_grad=True)
    mha = MultiHeadAttention(d_model=d_model, num_heads=h, dropout=0.0)
    out = mha(x_q=x, x_kv=x, mask=None)
    assert out.shape == (B, L, d_model)
    loss = out.pow(2).mean()
    loss.backward()
    grads_ok = all((p.grad is not None) for p in mha.parameters() if p.requires_grad)
    assert grads_ok, "Algún parámetro de MHA no recibió gradiente"

