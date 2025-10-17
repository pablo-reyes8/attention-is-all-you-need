import matplotlib.pyplot as plt
import torch 
import numpy as np

def positional_encoding(positions, d_model):
    """
    Computes sinusoidal positional encodings.
    Returns: tensor of shape (1, positions, d_model)
    """
    # posiciones (pos, 1)
    pos = np.arange(positions)[:, np.newaxis]
    # dimensiones (1, d/2)
    i = np.arange(0, d_model, 2)
    div = np.exp(i * (-np.log(10000.0) / d_model))
    angles = pos * div

    pe = np.zeros((positions, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)

    pe = pe[np.newaxis, ...]  # (1, positions, d_model)
    return torch.tensor(pe, dtype=torch.float32)



def plot_positional_encoding(pe: torch.Tensor, title="Positional Encoding"):
    """
    pe: (1, L, d) tensor
    Visualización mejorada: límites simétricos y ticks legibles.
    """
    pe_np = pe.squeeze(0).cpu().numpy()
    L, D = pe_np.shape
    vmax = 1.0
    vmin = -1.0

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pe_np, aspect="auto", origin="upper",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("Dimensión d")
    ax.set_ylabel("Posición")
    ax.set_title(title)
    ax.set_xticks(np.linspace(0, D-1, num=min(10, D), dtype=int))
    ax.set_yticks(np.linspace(0, L-1, num=min(10, L), dtype=int))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Valor seno/cos")
    plt.tight_layout()
    plt.show()