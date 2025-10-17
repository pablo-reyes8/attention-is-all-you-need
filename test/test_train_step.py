import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from model.transformer import *

PAD_ID = 0

def make_padding_mask(ids): 
    return (ids == PAD_ID)

def make_look_ahead_mask_T(T, device=None):
    return torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)

def combine_decoder_masks(tgt_in, la_TT):
    B, T = tgt_in.shape
    return la_TT[None, :, :] | (tgt_in == PAD_ID)[:, None, :]

def small_cfg():
    return dict(d_model=128, num_layers=2, num_heads=4, d_ff=512,
                vocab_size=256, max_pos=64)

def test_transformer_train_step(device):
    cfg = small_cfg()
    model = Transformer(
        num_layers=cfg["num_layers"], d_model=cfg["d_model"], num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        src_vocab_size=cfg["vocab_size"], tgt_vocab_size=cfg["vocab_size"],
        max_pos_src=cfg["max_pos"], max_pos_tgt=cfg["max_pos"],
        dropout=0.1, layernorm_eps=1e-6).to(device)
    model.train()

    B, Ls, Lt = 4, 20, 16
    src = torch.randint(4, cfg["vocab_size"], (B, Ls), device=device)
    tgt = torch.randint(4, cfg["vocab_size"], (B, Lt), device=device)
    src[:, -2:] = PAD_ID; tgt[:, -2:] = PAD_ID
    la = make_look_ahead_mask_T(Lt, device=device)

    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    src_pad = make_padding_mask(src)
    dec_mask = combine_decoder_masks(tgt_in, la[:-1, :-1])

    logits, _ = model(src_ids=src, tgt_ids_in=tgt_in,
                      src_key_padding_mask=src_pad,
                      look_ahead_mask=dec_mask,
                      enc_padding_mask=src_pad, return_attn=False)
    Bz, Tz, V = logits.shape
    assert V == cfg["vocab_size"]

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    loss = loss_fn(logits.reshape(Bz*Tz, V), tgt_out.reshape(Bz*Tz))
    assert torch.isfinite(loss)

    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    opt.step()