
import torch
import torch.nn.functional as F
import torch.nn as nn
from model.architecture.encoder import * 
from model.architecture.decoder import * 

class Transformer(nn.Module):
    """
    Encoder-Decoder Transformer:
      - encoder: (src_ids, src_key_padding_mask)
      - decoder: (tgt_ids_in, enc_out, look_ahead_mask, enc_padding_mask)
      - final linear -> logits (sin softmax)
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 src_vocab_size: int, tgt_vocab_size: int,
                 max_pos_src: int, max_pos_tgt: int,
                 dropout: float = 0.1, layernorm_eps: float = 1e-6 , weight_tying = False):

        super().__init__()
        self.encoder = Encoder(num_layers=num_layers,d_model=d_model,
                               num_heads=num_heads,
                               d_ff=d_ff,
                               vocab_size=src_vocab_size,
                               max_pos=max_pos_src,
                               dropout=dropout, layernorm_eps=layernorm_eps)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads,
                               d_ff=d_ff,
                               target_vocab_size=tgt_vocab_size,
                               max_pos=max_pos_tgt,
                               dropout=dropout,layernorm_eps=layernorm_eps)

        self.final_linear = nn.Linear(d_model, tgt_vocab_size)

        if weight_tying:
          self.final_linear.weight = self.decoder.embedding.weight

    def forward(self, src_ids, tgt_ids_in,
                src_key_padding_mask: torch.Tensor | None = None,
                look_ahead_mask: torch.Tensor | None = None,
                enc_padding_mask: torch.Tensor | None = None,
                return_attn: bool = True):
        """
        Returns:
          logits: (B, L_tgt, vocab_tgt)
          attn_weights: dict (o None)
        """

        enc_out = self.encoder(src_ids, src_key_padding_mask=src_key_padding_mask)

        dec_out, attn_weights = self.decoder(tgt_ids_in, enc_out, look_ahead_mask=look_ahead_mask,
                                             enc_padding_mask=enc_padding_mask,
                                             return_attn=return_attn)

        logits = self.final_linear(dec_out)  # (B,L_tgt,V_tgt) sin softmax

        if self.final_linear.weight.data_ptr() == self.decoder.embedding.weight.data_ptr():
          logits = logits / math.sqrt(self.decoder.d_model)

        return logits, attn_weights