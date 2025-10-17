# Attention Is All You Need — PyTorch Reproduction (EN↔ES MT)

A compact, reproduction of the original **Transformer** architecture in PyTorch, tailored for **English↔Spanish machine translation** using the OPUS‑100 dataset and **SentencePiece** BPE. This repository emphasizes clarity, correctness, and testability—aimed at students and researchers who want a clean, minimal, and modern baseline that trains on a single T4 GPU.

---

## ✨ Highlights

- **Faithful encoder–decoder** with **post-norm** sublayers (LayerNorm after residual).
- **Sinusoidal positional encoding** (`register_buffer`), deterministic and unit-tested.
- **Clean dataloading** for OPUS-100 (EN–ES) + **SentencePiece** BPE (shared vocab).
- **Robust masks**: padding + causal look-ahead, broadcast-safe to `(B, H, Lq, Lk)`.
- **Modern training**: Adam(β1=0.9, β2=0.98, eps=1e-9), **Noam LR** (warmup in *steps*), label smoothing, grad clipping.
- **In-loop metrics (no decoding)**: token accuracy and perplexity (PPL = `exp(loss)`).
- **Teacher-forced preview**: quick *argmax* snapshots to sniff quality during training.
- **Optional weight tying**: `tgt_embed ↔ softmax` (+ optional `src↔tgt` if shared vocab) with **correct logits scaling**.
- **Unit tests** (pytest) for PE, masks, MHA, Encoder/Decoder, a full train step, and logits shapes.

---

## 🧱 Architecture

- **Token embeddings** (learned end-to-end) + **sinusoidal PosEnc**.  
  *Note:* only the rows (tokens) used in a batch receive gradients/updates.
- **Multi-Head Attention** (scaled dot-product) with trainable **Wq, Wk, Wv, Wo**.
- **Position-wise FFN**: `Linear(d_model → d_ff)` → ReLU → `Linear(d_ff → d_model)` (default `d_ff = 4 × d_model`).
- **Residual + LayerNorm** (**post-norm**) in each sub-block.
- **Final linear to vocab** (no softmax in `forward`; use `CrossEntropyLoss` on logits for numerical stability).

> We intentionally **do not** apply softmax in the model forward; use `CrossEntropyLoss` on logits for numerical stability.

---

## 📦 Dataset & Tokenization

- **Dataset**: [`Helsinki-NLP/opus-100`](https://huggingface.co/datasets/Helsinki-NLP/opus-100), language pair **en–es**.
- **SentencePiece BPE** with **shared vocabulary** (default `vocab_size=16000`).
  - Special tokens: `pad_id=0`, `bos_id=1`, `eos_id=2`, `unk_id=3`.
- **Encoding:** `BOS + tokens + EOS`, `max_len` (e.g., `128` or `256`).
- **Collate & masks:**
  - **Key padding mask** `(B, Lk)` → expanded to `(B, 1, 1, Lk)`.
  - **Decoder look-ahead** (triangular causal) possibly combined with tgt padding → `(B, 1, Lq, Lk)`.
  - All masks **broadcastable** to `(B, H, Lq, Lk)`.

---

## ⚙️ Default Configuration (Tiny)

```python
d_model   = 256      # embedding width (and model width)
num_layers= 4        # encoder and decoder layers
num_heads = 4        # attention heads (d_head = d_model // num_heads)
d_ff     = 1024      # FFN hidden size (≈ 4 × d_model)
vocab    = 16000     # SentencePiece shared vocab
max_pos  = 256       # max positions for PE (>= max sequence length)
```

This configuration fits **comfortably on a T4** with a small subset of OPUS‑100 (e.g., 50k sentence pairs).


### Training

- **Scheduler:** Noam (Vaswani et al. 2017)

$$
lr(\text{step}) = d_{\text{model}}^{0.5} \min(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup}^{-1.5})
$$

`warmup` is in steps (batches), not epochs.

- **Recommended for 50k pairs, batch=64:**
  - steps_per_epoch ≈ 50,000 / 64 ≈ 782
  - warmup ≈ 1,600 (≈ 2 × steps/epoch)
  - log_every ≈ 80 (≈ 10 logs/epoch)
  - epochs = 10–20

- **Other:** gradient clipping (e.g., 1.0), AMP (mixed precision) recommended on T4.

### In-loop preview (teacher forcing):
Optionally print `argmax` predictions vs reference to monitor progress without decoding:

```python
HYP: ...  # model argmax under teacher forcing
REF: ...
```

## 📐 Transformer Math & Complexity (Quick)

### Scaled Dot-Product Attention
Given $Q\in\mathbb{R}^{L_q\times d_k}$, $K\in\mathbb{R}^{L_k\times d_k}$, $V\in\mathbb{R}^{L_k\times d_v}$:

$$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$$

Time (single head): $O(L_qL_k d_k)$; Memory: $O(L_qL_k)$ for attention weights.

### Multi-Head Attention (MHA)

$$\mathrm{MHA}(X)=\mathrm{Concat}(H_1,\dots,H_h)W_O,\quad H_i=\mathrm{Attn}(XW_i^Q,XW_i^K,XW_i^V)$$

With $d_h=d/h$: Params $\approx 4d^2$ (independent of $h$).  

Self-attn time ( $L_q=L_k=L$ ): $O(L^2 d + L d^2)$; Memory: $O(L^2)$.

### Position-wise FFN

$$\mathrm{FFN}(x)=\max(0,\,xW_1+b_1)W_2+b_2$$

Params $\approx 2 d\,d_{ff}$; Time $O(L\,d\,d_{ff})$; Activations $O(L\,d_{ff})$.

### Encoder Layer (post-norm)
1) $X\leftarrow \mathrm{LN}(X+\mathrm{MHA}(X))$  
2) $X\leftarrow \mathrm{LN}(X+\mathrm{FFN}(X))$

Per-layer time: $O(L^2 d + L d^2 + L d d_{ff})$ (bottleneck: $L^2 d$).

### Decoder Layer (post-norm)
Masked self-attn on $T$: $O(T^2 d)$; cross-attn to encoder length $S$: $O(T S d)$; FFN: $O(T d d_{ff})$.

### Masks (convention)
- **Padding:** boolean $True$ = block `<pad>`; broadcast to $(B,1,L_q,L_k)$.  
- **Look-ahead:** upper-triangular $[T\times T]$, $True$ = block future.  
- **Decoder self-attn:** combine as $M_{\text{look}} \lor M_{\text{pad-tgt}}$.

**Tips:** To fit on small GPUs, reduce $L$, $d$, or $d_{ff}$, or batch size. Use Noam schedule and label smoothing for stability.




---

## 🧪 Tests (pytest‑style)

The repository includes **self‑contained, assert‑based tests** grouped by module:

```
tests/
  conftest.py
  test_positional_encoding.py
  test_masks.py
  test_mha.py
  test_encoder_decoder.py
  test_train_step.py
```

**What we test:**
- Positional encoding shapes/values and numerics.
- Mask creation and combination (padding + look‑ahead).
- MHA shape contracts and gradient flow.
- Encoder/Decoder forward passes (finite outputs).
- A full training step: forward → loss → backward → optimizer step.

Run all tests:
```bash 
pytest -q
```

> If you don’t use pytest, you can still run the test functions directly; they are standard Python `assert` checks.

---

## 📁 Project Structure (suggested)

```
attention-is-all-you-need/
│
├─ data/                            # Data loading & preprocessing
│   ├─ load_data.py                 # Dataset loading (HuggingFace / OPUS-100)
│   └─ evaluate_data.py             # Corpus sanity checks, length stats, etc.
│
├─ model/
│   ├─ architecture/                # Core Transformer components
│   │   ├─ __init__.py              # Reexports for cleaner imports
│   │   ├─ attention.py             # Multi-Head Attention (Q, K, V, scaled dot-product)
│   │   ├─ encoder.py               # Encoder block (Self-Attention + FFN)
│   │   ├─ decoder.py               # Decoder block (Masked MHA + Cross-Attention)
│   │   ├─ pos_encoding.py          # Positional Encoding (sinusoidal)
│   │   └─ transformer_masks.py     # Padding & look-ahead masks
│   │
│   ├─ training/                    # Training utilities and loop
│   │   ├─ training_loop.py         # Full training step (forward + backward)
│   │   ├─ training_utils.py        # LR schedulers, label smoothing, clipping
│   │   └─ transformer_utils.py     # Weight initialization, parameter counting
│   │
│   ├─ transformer.py               # High-level Encoder–Decoder assembly
│
├─ test/                            # Unit tests (pytest style)
│   ├─ conftest.py                  # Pytest fixtures (mocks, sample tensors)
│   ├─ test_mha.py                  # Tests for Multi-Head Attention
│   ├─ test_positional_encoding.py  # PE correctness and shape
│   ├─ test_masks.py                # Mask logic (padding & look-ahead)
│   ├─ test_encoder_decoder.py      # Encoder/Decoder forward consistency
│   └─ test_train_step.py           # Gradient flow, optimizer updates
│
├─ training_showcase/               # Example notebooks
│   └─ training_showcase.ipynb      # Interactive training + evaluation demo
│
├─ README.md                        # Project overview, setup, usage
```

---

## ✅ Tips & Notes

- If you see early training instability, increase `warmup` (e.g., 6000–8000).
- To reduce memory: lower `batch_size`, `d_model`, or `num_layers`.
- Keep `d_model` divisible by `num_heads`.
- Do **not** apply softmax in the model; use CE on logits.
- Label smoothing (0.1) can help generalization on small corpora.

---

## 📚 References

- Vaswani et al., *Attention Is All You Need*, 2017.  
- OPUS‑100: A Multi-Lingual Dataset for Machine Translation.  
- SentencePiece: *A simple and language independent subword tokenizer and detokenizer*.

---

## 📜 License

MIT License — see `LICENSE` for details.
