import torch

PAD_ID = 0

def make_padding_mask(ids: torch.Tensor, pad_id: int = PAD_ID):
    """
    ids: (B, L)
    return: (B, L) bool  True = BLOQUEAR (son pads)
    """
    return (ids == pad_id)

def make_look_ahead_mask(T: int, device=None):
    """
    return: (T, T) bool  True = BLOQUEAR (futuro)
    """
    m = torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)
    return m

def combine_decoder_masks(tgt_in: torch.Tensor, look_ahead: torch.Tensor, pad_id: int = PAD_ID):
    """
    tgt_in: (B, T)  entradas del decoder (p.ej. tgt[:, :-1])
    look_ahead: (T, T) bool
    return: (B, T, T) bool  True = BLOQUEAR
    """
    B, T = tgt_in.shape
    tgt_pad = make_padding_mask(tgt_in, pad_id)
    # expandir padding a (B, 1, T) para actuar sobre K/V, y broadcast sobre Lq
    tgt_pad_exp = tgt_pad[:, None, :]
    # look-ahead a (1, T, T) para broadcast en batch
    la = look_ahead[None, :, :]
    # combinar (OR): se bloquea si es futuro o es pad en K/V
    # al hacer atención, forma destino efectiva será (B, 1, Tq, Tk) por broadcast
    combined = la | tgt_pad_exp
    return combined


