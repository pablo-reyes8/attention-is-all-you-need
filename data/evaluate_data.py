import numpy as np
import torch


def check_dataset_sizes(ds_tok):
    """
    Imprime el tamaño de cada split tokenizado (train/validation/test).

    Parameters
    ----------
    ds_tok : dict or DatasetDict
        Dataset tokenizado con splits.
    """
    print("Tamaño del corpus OPUS-100 (EN↔ES):")
    for split in ds_tok:
        size = len(ds_tok[split])
        print(f"  {split:12s} → {size:,} pares de oraciones")


def check_subset(ds_tok, n_subset=50_000, seed=42):
    """
    Crea y muestra información del subconjunto de entrenamiento.

    Parameters
    ----------
    ds_tok : DatasetDict
        Dataset tokenizado.
    n_subset : int
        Número de ejemplos para el subconjunto.
    seed : int
        Semilla de barajado.

    Returns
    -------
    ds_small : Dataset
        Subconjunto seleccionado.
    """
    ds_small = ds_tok["train"].shuffle(seed=seed).select(range(n_subset))
    print(f"\n Subconjunto de entrenamiento: {len(ds_small):,} pares (≈{100*n_subset/len(ds_tok['train']):.1f}%)")
    return ds_small


def show_random_example(ds_small, sp):
    """
    Muestra un par ejemplo (en → es) decodificado con el tokenizer.

    Parameters
    ----------
    ds_small : Dataset
        Subconjunto reducido del split de entrenamiento.
    sp : sentencepiece.SentencePieceProcessor
        Tokenizador SentencePiece entrenado.
    """
    ex = ds_small[0]
    print("\n Ejemplo de par (en → es):")
    print("EN:", sp.decode(ex["src"]))
    print("ES:", sp.decode(ex["tgt"]))


def check_average_lengths(ds_raw, sample_size=2000):
    """
    Calcula la longitud promedio de las oraciones EN y ES (en palabras)
    para verificar balance lingüístico.

    Parameters
    ----------
    ds_raw : DatasetDict
        Dataset original sin tokenizar (con claves 'translation').
    sample_size : int
        Tamaño de la muestra a analizar.
    """
    sample = ds_raw["train"].select(range(sample_size))
    lens_en = [len(ex["translation"]["en"].split()) for ex in sample]
    lens_es = [len(ex["translation"]["es"].split()) for ex in sample]
    print(f"\n Longitud promedio (EN): {np.mean(lens_en):.1f} palabras")
    print(f"Longitud promedio (ES): {np.mean(lens_es):.1f} palabras")


def check_batch_shapes(loader):
    """
    Toma un batch de un DataLoader y muestra las dimensiones y tipos
    de tensores, útil para detectar errores en padding/máscaras.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        DataLoader de entrenamiento o validación.
    """
    batch = next(iter(loader))
    print("\nSanity check del batch:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:12s} → {tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"{k:12s} → {type(v)}")


def check_padding_consistency(loader, pad_id):
    """
    Verifica que el padding se haya aplicado correctamente y que los tokens
    de padding correspondan al `pad_id` configurado.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        DataLoader a testear.
    pad_id : int
        ID del token de padding (sp.pad_id()).
    """
    batch = next(iter(loader))
    src_pad = (batch["src"] == pad_id).float().mean().item()
    tgt_pad = (batch["tgt"] == pad_id).float().mean().item()
    print(f"\n Porcentaje de padding:")
    print(f"  SRC padding: {src_pad*100:.2f}%")
    print(f"  TGT padding: {tgt_pad*100:.2f}%")



def run_all_checks(ds_raw, ds_tok, sp, loaders, subset_size=50_000):
    """
    Ejecuta todos los sanity checks básicos en orden lógico.

    Parameters
    ----------
    ds_raw : DatasetDict
        Dataset original sin tokenizar.
    ds_tok : DatasetDict
        Dataset tokenizado.
    sp : sentencepiece.SentencePieceProcessor
        Tokenizador SentencePiece.
    loaders : dict
        Diccionario con DataLoaders {"train", "validation", "test"}.
    subset_size : int
        Tamaño del subconjunto de entrenamiento.
    """
    check_dataset_sizes(ds_tok)
    ds_small = check_subset(ds_tok, n_subset=subset_size)
    show_random_example(ds_small, sp)
    check_average_lengths(ds_raw)
    check_batch_shapes(loaders["train"])
    check_padding_consistency(loaders["train"], pad_id=sp.pad_id())
    print("\nTodos los sanity checks completados correctamente.\n")

