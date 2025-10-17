import torch

PAD_ID = 0

def make_padding_mask(ids: torch.Tensor, pad_id: int = PAD_ID):
    return (ids == pad_id)

def make_look_ahead_mask_T(T: int, device=None):
    return torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)

def combine_decoder_masks(tgt_in: torch.Tensor, look_ahead_TT: torch.Tensor):
    B, T = tgt_in.shape
    la = look_ahead_TT[None, :, :]
    tgt_pad = (tgt_in == PAD_ID)[:, None, :]
    return la | tgt_pad

def test_look_ahead_mask():
    T = 10
    m = make_look_ahead_mask_T(T)
    assert m.shape == (T, T)
    assert (m.triu(1) == True).all()
    assert (torch.diag(m.diag()) == False).all()
    assert (torch.tril(m) == False).all()

def test_combine_decoder_masks_blocks_pads_and_future():
    B, T = 2, 8
    tgt_in = torch.randint(4, 100, (B, T))
    tgt_in[:, -2:] = PAD_ID
    la = make_look_ahead_mask_T(T)
    comb = combine_decoder_masks(tgt_in, la)
    assert comb.shape == (B, T, T)
    # columnas K en pad bloqueadas
    assert comb[:, :, -1].all() and comb[:, :, -2].all()



