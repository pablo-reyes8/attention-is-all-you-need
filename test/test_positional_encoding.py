import torch
from model.architecture.pos_encoding import * 

def test_positional_encoding_basic():
    max_pos, d_model = 32, 16
    pe = positional_encoding(max_pos, d_model)
    assert pe.shape == (1, max_pos, d_model)
    assert pe.dtype == torch.float32
    row0 = pe[0, 0]
    assert torch.allclose(row0[0::2], torch.zeros_like(row0[0::2]), atol=1e-6)
    assert torch.allclose(row0[1::2], torch.ones_like(row0[1::2]), atol=1e-6)
    assert torch.isfinite(pe).all()
    assert pe.abs().max() <= 1.0 + 1e-5

