import torch 
import torch.nn.functional as F
import torch.nn as nn
from model.training.training_utils import *
from torch.nn.utils import clip_grad_norm_
import math
from model.architecture.transformer_masks import *
PAD_ID = 0


def train_transformer_mt(
    model,
    train_loader,
    val_loader=None,
    *,
    d_model=256, # El tama;o de la dimension de los embeddings por token
    epochs=10,
    base_lr=1.0,          # Noam escalará esto
    warmup=1600, # Se mide en steps no en epcohs
    label_smoothing=0.1,  # smoothing sólo en train
    grad_clip=1.0,
    device="cuda",
    ckpt_path="transformer_best.pt",
    log_every=80 , preview_every=None , # e.g., 100 -> imprime ejemplo cada 100 iters; None = desactivado
    id2tok_fn=None ):   # callable: List[int] -> str, para decodificar ids a texto

    """
    Entrena un Transformer seq2seq (encoder-decoder) para traducción.
    - Usa CrossEntropy con ignore_index=PAD_ID
    - Scheduler Noam (paper)
    - Gradient clipping
    - Máscaras: padding (src) + look-ahead y padding (tgt)
    - Guarda el mejor checkpoint por val_loss (si val_loader no es None)
    """
    device = torch.device(device)
    torch.set_float32_matmul_precision("high")
    model.to(device)
    model.train()


    # DEFINIMOS LOS HIPERPARAMEROS DENTO DE LA FUNCION
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler= NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup)

    ce_train = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=label_smoothing)
    ce_eval = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.0)



    best_val = float("inf")
    history = { "train_loss": [], "val_loss": [],
        "train_ppl":  [], "val_ppl":  [],
        "train_tok_acc": [], "val_tok_acc": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        epoch_loss_sum, epoch_tokens = 0.0, 0
        epoch_correct, epoch_total   = 0, 0
        t0 = time.time()

        for it, batch in enumerate(train_loader, 1):
            src = batch["src"].to(device, non_blocking=True)
            tgt = batch["tgt"].to(device, non_blocking=True)
            la_TT = batch["tgt_mask"].to(device, non_blocking=True)

            # Teacher forcing
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:,  1:]

            # Máscaras
            src_pad_mask = make_padding_mask(src)
            look_ahead = la_TT[:-1, :-1]
            dec_mask = combine_decoder_masks(tgt_in, look_ahead)

            # Forward
            logits, _ = model(src_ids= src,
                              tgt_ids_in= tgt_in,
                              src_key_padding_mask= src_pad_mask,
                              look_ahead_mask= dec_mask,
                              enc_padding_mask= src_pad_mask,
                              return_attn=False)            # (B, T, V)

            B, T, V = logits.shape
            loss = ce_train(logits.reshape(B * T, V), tgt_out.reshape(B * T))

            optimizer.zero_grad(set_to_none=True)
            loss.backward() # Retropropagacion

            if grad_clip is not None:
                clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()

            # Ponderación por tokens no-PAD
            nonpad = (tgt_out != PAD_ID).sum().item()
            epoch_loss_sum += loss.item() * nonpad
            epoch_tokens += nonpad

            #token accuracy
            corr, tot = token_acc(logits, tgt_out)
            epoch_correct += corr
            epoch_total   += tot

            # Logging periódico + preview opcional de “traducción” bajo teacher forcing
            if it % log_every == 0:
                tok_per_sec = epoch_tokens / (time.time() - t0 + 1e-9)
                avg_loss = epoch_loss_sum / max(1, epoch_tokens)
                avg_ppl  = math.exp(avg_loss)
                avg_acc  = (epoch_correct / max(1, epoch_total)) * 100.0
                print(f"[Epoch {epoch} | {it:4d}/{len(train_loader)}] "
                      f"train_loss={avg_loss:.4f}  ppl={avg_ppl:.2f}  "
                      f"tok_acc={avg_acc:.2f}%  tok/s={tok_per_sec:,.0f}")

            # Preview opcional (teacher forcing): argmax sobre logits vs tgt_out
            if (preview_every is not None) and (id2tok_fn is not None) and (it % preview_every == 0):
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)  # (B, T)
                    b0 = 0
                    cut_pred = preds[b0].tolist()
                    cut_tout = tgt_out[b0].tolist()

                    if PAD_ID in cut_tout:
                        L = cut_tout.index(PAD_ID)
                        cut_pred = cut_pred[:L]
                        cut_tout = cut_tout[:L]

                    hyp = id2tok_fn(cut_pred)  # str
                    ref = id2tok_fn(cut_tout)  # str
                    print("— preview (teacher-forced argmax) —")
                    print("HYP:", hyp)
                    print("REF:", ref)

        train_loss = epoch_loss_sum / max(1, epoch_tokens)
        train_ppl  = math.exp(train_loss)
        train_acc  = (epoch_correct / max(1, epoch_total)) * 100.0
        history["train_loss"].append(train_loss)
        history["train_ppl"].append(train_ppl)
        history["train_tok_acc"].append(train_acc)


        #  EVAL (Solo si se activa)
        if val_loader is None:
            print(f"Epoch {epoch} done | train_loss={train_loss:.4f}")
            continue

        model.eval()
        val_loss_sum, val_tokens = 0.0, 0
        val_correct, val_total   = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                src = batch["src"].to(device, non_blocking=True)
                tgt = batch["tgt"].to(device, non_blocking=True)
                la_TT = batch["tgt_mask"].to(device, non_blocking=True)

                tgt_in  = tgt[:, :-1]
                tgt_out = tgt[:,  1:]

                src_pad_mask = make_padding_mask(src)
                look_ahead = la_TT[:-1, :-1]
                dec_mask = combine_decoder_masks(tgt_in, look_ahead)

                logits, _ = model(src_ids=src,
                                  tgt_ids_in=tgt_in,
                                  src_key_padding_mask=src_pad_mask,
                                  look_ahead_mask=dec_mask,
                                  enc_padding_mask=src_pad_mask,
                                  return_attn=False)

                B, T, V = logits.shape
                loss = ce_eval(logits.reshape(B * T, V), tgt_out.reshape(B * T))

                nonpad = (tgt_out != PAD_ID).sum().item()
                val_loss_sum += loss.item() * nonpad
                val_tokens   += nonpad

                # Métrica: token accuracy (eval)
                corr, tot = token_acc(logits, tgt_out)
                val_correct += corr
                val_total   += tot

        val_loss = val_loss_sum / max(1, val_tokens)
        val_ppl  = math.exp(val_loss)
        val_acc  = (val_correct / max(1, val_total)) * 100.0

        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["val_tok_acc"].append(val_acc)

        print(f"Epoch {epoch} done | "f"train_loss={train_loss:.4f}  train_ppl={train_ppl:.2f}  train_tok_acc={train_acc:.2f}%  "
              f"val_loss={val_loss:.4f}      val_ppl={val_ppl:.2f}      val_tok_acc={val_acc:.2f}%")

        # Guardar mejor checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss}, ckpt_path)
            print(f"✓ Guardado checkpoint (best val_loss={val_loss:.4f}) -> {ckpt_path}")

    return history