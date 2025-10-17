import torch
from model.architecture.encoder import * 
from model.architecture.decoder import *


PAD_ID = 0

def make_padding_mask(ids: torch.Tensor):
    return (ids == PAD_ID)


def make_look_ahead_mask_T(T: int, device=None):
    return torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)


def combine_decoder_masks(tgt_in: torch.Tensor, la_TT: torch.Tensor):
    B, T = tgt_in.shape
    return la_TT[None, :, :] | (tgt_in == PAD_ID)[:, None, :]


def test_encoder_forward_no_nan(cfg):
    B, Ls = 2, 24
    enc = Encoder(num_layers=cfg["num_layers"], d_model=cfg["d_model"],
                  num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
                  vocab_size=cfg["vocab_size"], max_pos=cfg["max_pos"])
    src = torch.randint(4, cfg["vocab_size"], (B, Ls))
    src[:, -2:] = PAD_ID
    mask = make_padding_mask(src)
    out = enc(src, src_key_padding_mask=mask)
    assert out.shape == (B, Ls, cfg["d_model"])
    assert torch.isfinite(out).all()


def test_decoder_forward_no_nan(cfg):
    B, Ls, Lt = 2, 20, 14
    enc = Encoder(num_layers=cfg["num_layers"], d_model=cfg["d_model"],
                  num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
                  vocab_size=cfg["vocab_size"], max_pos=cfg["max_pos"])
    dec = Decoder(num_layers=cfg["num_layers"], d_model=cfg["d_model"],
                  num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
                  target_vocab_size=cfg["vocab_size"], max_pos=cfg["max_pos"])
    src = torch.randint(4, cfg["vocab_size"], (B, Ls))
    tgt = torch.randint(4, cfg["vocab_size"], (B, Lt))
    src[:, -2:] = PAD_ID; tgt[:, -2:] = PAD_ID
    la = make_look_ahead_mask_T(Lt)
    enc_out = enc(src, src_key_padding_mask=make_padding_mask(src))
    tgt_in = tgt[:, :-1]
    dec_mask = combine_decoder_masks(tgt_in, la[:-1, :-1])
    out, _ = dec(tgt_in, enc_out, look_ahead_mask=dec_mask,
                 enc_padding_mask=make_padding_mask(src), return_attn=False)
    assert out.shape == (B, Lt-1, cfg["d_model"])
    assert torch.isfinite(out).all()