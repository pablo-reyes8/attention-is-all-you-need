import torch
import pytest

from model.architecture.attention import * 
from model.architecture.decoder import * 
from model.architecture.pos_encoding import * 
from model.architecture.transformer_masks import * 

from model.transformer import *


PAD_ID = 0

def small_cfg():
    return dict(d_model=128, num_layers=2, num_heads=4, d_ff=512,
                vocab_size=256, max_pos=64)

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture()
def cfg():
    return small_cfg()

@pytest.fixture()
def model_tiny(cfg, device):
    m = Transformer(
        num_layers=cfg["num_layers"], d_model=cfg["d_model"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        src_vocab_size=cfg["vocab_size"], tgt_vocab_size=cfg["vocab_size"],
        max_pos_src=cfg["max_pos"], max_pos_tgt=cfg["max_pos"],
        dropout=0.1, layernorm_eps=1e-6
    ).to(device)
    return m

@pytest.fixture()
def fake_batch(cfg, device):
    B, Ls, Lt = 4, 20, 16
    src = torch.randint(4, cfg["vocab_size"], (B, Ls), device=device)
    tgt = torch.randint(4, cfg["vocab_size"], (B, Lt), device=device)
    src[:, 0] = 1; src[:, -1] = 2; tgt[:, 0] = 1; tgt[:, -1] = 2
    src[:, -2:] = PAD_ID; tgt[:, -2:] = PAD_ID
    causal = torch.triu(torch.ones((Lt, Lt), dtype=torch.bool, device=device), diagonal=1)

    return {"src": src, "tgt": tgt, "tgt_mask": causal}



