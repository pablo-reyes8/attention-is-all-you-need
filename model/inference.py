"""Command-line helper for running Transformer inference.

This module loads a trained checkpoint together with the SentencePiece
tokenizer used during training and performs greedy decoding to translate
English sentences into Spanish.  It mirrors the masking logic employed
during training so that the decoder receives the same look-ahead and
padding masks that it expects.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import torch
import sentencepiece as spm

from model.transformer import Transformer
from model.architecture.transformer_masks import (
    make_padding_mask,
    make_look_ahead_mask,
    combine_decoder_masks,
)


def load_sentencepiece_model(model_path: str | Path):
    """Load a SentencePiece tokenizer from ``model_path``.

    Parameters
    ----------
    model_path:
        Path to the ``.model`` file produced by SentencePiece.
    """

    processor = spm.SentencePieceProcessor()
    if not processor.Load(str(model_path)):
        raise ValueError(f"Unable to load SentencePiece model from {model_path!s}")
    return processor


def build_model(sp: spm.SentencePieceProcessor, args: argparse.Namespace):
    """Instantiate the Transformer architecture for inference."""

    vocab_size = sp.get_piece_size()

    model = Transformer(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        src_vocab_size=args.src_vocab_size or vocab_size,
        tgt_vocab_size=args.tgt_vocab_size or vocab_size,
        max_pos_src=args.max_pos_src,
        max_pos_tgt=args.max_pos_tgt,
        dropout=args.dropout,
        layernorm_eps=args.layernorm_eps,
        weight_tying=args.weight_tying,)
    
    return model


def load_checkpoint(model: Transformer, checkpoint_path: str | Path, device: torch.device) :
    """Load weights from ``checkpoint_path`` into ``model``."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)


@torch.inference_mode()
def greedy_translate(
    model: Transformer,
    tokenizer: spm.SentencePieceProcessor,
    text: str,
    *,
    max_length: int,
    device: torch.device,):

    """Translate ``text`` (English) into Spanish using greedy decoding."""

    # Encode source sentence and move tensors to the desired device.
    src_ids = [tokenizer.bos_id()]
    src_ids.extend(tokenizer.encode(text, out_type=int))
    src_ids.append(tokenizer.eos_id())
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    pad_id = tokenizer.pad_id()
    src_padding_mask = make_padding_mask(src_tensor, pad_id=pad_id)

    # Encode source once.
    encoder_memory = model.encoder(src_tensor, src_key_padding_mask=src_padding_mask)

    # Decoder input starts with BOS.
    generated = torch.tensor([[tokenizer.bos_id()]], dtype=torch.long, device=device)

    for _ in range(max_length):
        look_ahead = make_look_ahead_mask(generated.size(1), device=device)
        decoder_mask = combine_decoder_masks(generated, look_ahead, pad_id=pad_id)

        decoder_output, _ = model.decoder(
            generated,
            encoder_memory,
            look_ahead_mask=decoder_mask,
            enc_padding_mask=src_padding_mask,
            return_attn=False)

        logits = model.final_linear(decoder_output)
        if model.final_linear.weight.data_ptr() == model.decoder.embedding.weight.data_ptr():
            logits = logits / math.sqrt(model.decoder.d_model)

        next_token = logits[:, -1, :].argmax(dim=-1)
        next_id = next_token.item()

        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        if next_id == tokenizer.eos_id():
            break

    out_ids = generated.squeeze(0).tolist()

    if out_ids and out_ids[0] == tokenizer.bos_id():
        out_ids = out_ids[1:]
    if out_ids and out_ids[-1] == tokenizer.eos_id():
        out_ids = out_ids[:-1]

    return tokenizer.decode(out_ids)


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="Run Transformer inference for EN→ES translation")
    parser.add_argument("checkpoint", help="Path to the trained Transformer checkpoint (.pt)")
    parser.add_argument("sp_model", help="Path to the SentencePiece .model file used during training")
    parser.add_argument("text", nargs="?", help="Sentence in English to translate. If omitted, read from stdin.")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (default: auto-detect)")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of encoder/decoder layers")
    parser.add_argument("--d-model", type=int, default=256, help="Transformer model width")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=1024, help="Hidden size of the feed-forward network")
    parser.add_argument("--src-vocab-size", type=int, default=0,
                        help="Source vocabulary size (default: use SentencePiece size)")
    parser.add_argument("--tgt-vocab-size", type=int, default=0,
                        help="Target vocabulary size (default: use SentencePiece size)")
    parser.add_argument("--max-pos-src", type=int, default=256, help="Maximum source positions for positional encoding")
    parser.add_argument("--max-pos-tgt", type=int, default=256, help="Maximum target positions for positional encoding")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used during training")
    parser.add_argument("--layernorm-eps", type=float, default=1e-6, help="LayerNorm epsilon value")
    parser.add_argument("--no-weight-tying", action="store_true",
                        help="Disable weight tying between decoder embeddings and softmax")
    parser.add_argument("--max-decode-len", type=int, default=128,
                        help="Maximum number of tokens to decode (excluding BOS)")

    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) :
    args = parse_args(argv)

    device = torch.device(args.device)
    weight_tying = not args.no_weight_tying
    setattr(args, "weight_tying", weight_tying)

    sp_model = load_sentencepiece_model(args.sp_model)
    model = build_model(sp_model, args)
    load_checkpoint(model, args.checkpoint, device)

    model.to(device)
    model.eval()

    if args.text is not None:
        sentences = [args.text]
    else:
        print("Enter English sentences (Ctrl-D to finish):")
        sentences = None

    if sentences is not None:
        iterable: Iterable[str] = sentences
    else:
        def _stdin_iter() -> Iterable[str]:
            while True:
                try:
                    line = input("› ")
                except EOFError:
                    break
                stripped = line.strip()
                if stripped:
                    yield stripped

        iterable = _stdin_iter()

    for sentence in iterable:
        translation = greedy_translate(
            model,
            sp_model,
            sentence,
            max_length=args.max_decode_len,
            device=device,
        )
        print(f"EN: {sentence}")
        print(f"ES: {translation}\n")


if __name__ == "__main__":
    main()
