import torch


def _subsequent_mask(sz: int):
    return torch.triu(torch.ones((sz, sz), dtype=torch.bool), diagonal=1)[None, :, :]

def _key_padding_mask(tokens: torch.Tensor, pad_id: int):
    return (tokens == pad_id)

@torch.no_grad()
def greedy_decode_en2es(
    model,
    text_en: str,
    sp_shared,              # SentencePiece compartido EN↔ES
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int = 128,
    device: str = "cuda"):
    """
    Traduce EN→ES con búsqueda codiciosa:
    forward(src_ids, tgt_ids_in, src_key_padding_mask, look_ahead_mask, enc_padding_mask, return_attn)
    Devuelve solo el texto en ES.
    """
    model.eval()

    src_ids = [bos_id] + sp_shared.encode(text_en, out_type=int)[:max_len-2] + [eos_id]
    src = torch.tensor(src_ids, dtype=torch.long, device=device)[None, :]   

    tgt = torch.tensor([[bos_id]], dtype=torch.long, device=device)        

    src_key_pad = _key_padding_mask(src, pad_id=pad_id)                  

    for _ in range(max_len - 1):  
        look_ahead = _subsequent_mask(tgt.size(1)).to(device)              

        logits, _ = model(
            src, tgt,
            src_key_padding_mask=src_key_pad,
            look_ahead_mask=look_ahead,
            enc_padding_mask=src_key_pad,
            return_attn=False) 

        next_token = torch.argmax(logits[:, -1, :], dim=-1)               
        next_id = next_token.item()
        tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)             

        if next_id == eos_id:
            break

    out = tgt[0].tolist()
    if out and out[0] == bos_id: out = out[1:]
    if out and out[-1] == eos_id: out = out[:-1]
    return sp_shared.decode(out)

def translate_en2es(model, text_en, sp, ids, device="cuda"):
    return greedy_decode_en2es(
        model, text_en, sp_shared=sp,
        bos_id=ids["bos"], eos_id=ids["eos"], pad_id=ids["pad"],
        device=device)

### EXAMPLE USE #### 

##### Setting the tokenizers #######

#tmp = tempfile.gettempdir()
#src_model = os.path.join(tmp, "spm_bpe.model")
#src_vocab = os.path.join(tmp, "spm_bpe.vocab")

#dest_dir = "./tokenizers" 
#os.makedirs(dest_dir, exist_ok=True)
#shutil.copy2(src_model, os.path.join(dest_dir, "spm_bpe.model"))
#shutil.copy2(src_vocab, os.path.join(dest_dir, "spm_bpe.vocab"))

#sp = spm.SentencePieceProcessor(model_file="./tokenizers/spm_bpe.model")
#PAD = sp.pad_id()   
#BOS = sp.bos_id()   
#EOS = sp.eos_id()   
#UNK = sp.unk_id()  

################### Inference ###################


#BOS, EOS, PAD = sp.bos_id(), sp.eos_id(), sp.pad_id()
#ids = {"bos": BOS, "eos": EOS, "pad": PAD}

#sent = "Que haces?"
#print(translate_en2es(model, sent, sp, ids, device="cuda"))