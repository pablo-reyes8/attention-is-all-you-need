import os
import tempfile
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import sentencepiece as spm


def build_mt_dataloaders(
    dataset_name: str = "Helsinki-NLP/opus-100",
    config_name: str = "en-es",
    src_lang: str = "en",
    tgt_lang: str = "es",*,
    vocab_size: int = 16_000,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    model_prefix = None,  
    bos_id: int = 1,
    eos_id: int = 2,
    pad_id: int = 0,
    unk_id: int = 3,
    max_len: int = 128,
    num_proc: int = 4,
    train_subset = 50_000, 
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,):

    """
    Construye DataLoaders para un problema de traducción usando OPUS-100, entrenando un
    tokenizer SentencePiece (BPE) a partir del split de entrenamiento y tokenizando
    (src, tgt) con padding dinámico y máscara causal para el decodificador.

    Parámetros
    ----------
    dataset_name : str
        Nombre del dataset en HuggingFace Datasets.
    config_name : str
        Configuración del dataset (por ejemplo, "en-es").
    src_lang : str
        Clave de idioma fuente dentro de `ex["translation"]` (e.g., "en").
    tgt_lang : str
        Clave de idioma objetivo dentro de `ex["translation"]` (e.g., "es").
    vocab_size : int
        Tamaño del vocabulario para SentencePiece.
    model_type : str
        Tipo de modelo SentencePiece ("bpe", "unigram", "char", "word").
    character_coverage : float
        Cobertura de caracteres para entrenamiento de SentencePiece.
    model_prefix : Optional[str]
        Prefijo (ruta) para guardar/recargar el tokenizer. Si None, usa `tempfile.gettempdir()`.
        Si ya existen `*.model` y `*.vocab` con ese prefijo, se reutilizan.
    bos_id, eos_id, pad_id, unk_id : int
        IDs para tokens especiales del tokenizer.
    max_len : int
        Longitud máxima (con BOS/EOS) que se conservará por secuencia (se trunca si excede).
    num_proc : int
        Paralelismo al mapear el dataset tokenizado.
    train_subset : Optional[int]
        Número de ejemplos a tomar del split de entrenamiento (tras barajar). Usa None para todo.
    seed : int
        Semilla para barajado y selección determinista del subconjunto de entrenamiento.
    batch_size : int
        Tamaño de lote para los DataLoaders.
    num_workers : int
        Número de *workers* para los DataLoaders.
    pin_memory : bool
        Si `True`, hace *pin* de memoria (útil cuando se entrena en GPU).

    Returns
    -------
    sp : sentencepiece.SentencePieceProcessor
        Tokenizer entrenado/cargado.
    loaders : Dict[str, DataLoader]
        Diccionario con `{"train": dl_train, "validation": dl_val, "test": dl_test}`.
    info : Dict[str, int]
        Metadatos útiles: `{"pad_id": ..., "bos_id": ..., "eos_id": ..., "unk_id": ..., "vocab_size": ...}`.

    Notas
    -----
    - El tokenizer se entrena concatenando líneas EN y ES del split de entrenamiento,
      lo que da un vocab compartido (útil para NMT).
    - El `collate_fn` hace padding dinámico y construye una máscara causal estricta
      (triangular superior) para el decodificador.
    - Este *pipeline* es agnóstico al modelo; puedes enchufarlo a un Transformer
      (encoder-decoder) o decoder-only con *teacher forcing*.
    """


    ds = load_dataset(dataset_name, config_name)

    if model_prefix is None:
        tmp_dir = tempfile.gettempdir()
        model_prefix = os.path.join(tmp_dir, "spm_bpe")

    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"

    if not (os.path.exists(model_file) and os.path.exists(vocab_file)):
        corpus_path = os.path.join(tempfile.gettempdir(), f"{config_name.replace('/','_')}_corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for ex in ds["train"]:
                f.write(ex["translation"][src_lang] + "\n")
                f.write(ex["translation"][tgt_lang] + "\n")

        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            unk_id=unk_id)

    sp = spm.SentencePieceProcessor(model_file=model_file)


    def _encode(ex):
        s = sp.encode(ex["translation"][src_lang], out_type=int)[: max_len - 2]
        t = sp.encode(ex["translation"][tgt_lang], out_type=int)[: max_len - 2]
        return {
            "src": [sp.bos_id()] + s + [sp.eos_id()],
            "tgt": [sp.bos_id()] + t + [sp.eos_id()],}

    cols = ds["train"].column_names
    ds_tok = ds.map(_encode, remove_columns=cols, num_proc=num_proc)

    # Subconjunto de entrenamiento (opcional)
    if train_subset is not None:
        ds_train = ds_tok["train"].shuffle(seed=seed).select(range(train_subset))
    else:
        ds_train = ds_tok["train"]

    ds_val = ds_tok["validation"]
    ds_test = ds_tok["test"]


    PAD = sp.pad_id()
    def _collate(batch):
        src = [torch.tensor(b["src"], dtype=torch.long) for b in batch]
        tgt = [torch.tensor(b["tgt"], dtype=torch.long) for b in batch]

        src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=PAD)
        tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=PAD)

        T = tgt.size(1)
        causal = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1)
        # Máscara de padding para src/tgt (útil para atención y pérdidas)
        src_pad_mask = (src == PAD)  
        tgt_pad_mask = (tgt == PAD)  #

        return {
            "src": src,        
            "tgt": tgt,                
            "tgt_mask": causal,       
            "src_pad_mask": src_pad_mask,
            "tgt_pad_mask": tgt_pad_mask,}


    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, num_workers=num_workers, pin_memory=pin_memory)
    
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=num_workers, pin_memory=pin_memory)
    
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=num_workers, pin_memory=pin_memory)

    loaders = {"train": train_loader, "validation": val_loader, "test": test_loader}
    info = {"pad_id": PAD,
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "unk_id": sp.unk_id(),
        "vocab_size": sp.get_piece_size()}
    
    return sp, loaders, info , ds , ds_tok