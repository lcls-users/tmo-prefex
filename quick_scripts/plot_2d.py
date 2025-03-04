# Plot a numpy array as a 2D heat map with labels.
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import typer

#ids = [  0,  22,  45, 112,
#       180, 337, 135, 157,
#        90, 202, 225, 247,
#       270,  67, 315, 292 ]
# transpose
ids = [  0, 180,  90, 270,
        22, 337, 202,  67,
        45, 135, 225, 315,
       112, 157, 247, 292]

def main(name: str,
         title: Optional[str] = None,
         norm: bool = False):
    C = np.load(f"{name}.npy")
    if norm:
        scale = ( np.diag(C) + (np.abs(np.diag(C)) <= 1e-8) )**-0.5
        C *= scale[:,None]*scale[None,:]
        C -= np.diag(np.diag(C))

    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(C)
    N = len(C)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(N), labels=ids)
    ax.set_yticks(np.arange(N), labels=ids)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    if C.max() > 1.0:
        for i in range(N):
            for j in range(i+1):
                text = ax.text(j, i, "%.0f"%(C[i, j]),
                               ha="center", va="center", color="w")
    else:
        fig.colorbar(im)
    for i in range(4, 16, 4):
        ax.axhline(i-0.5, color='w')
        ax.axvline(i-0.5, color='w')

    if title:
        ax.set_title(title)
    fig.tight_layout()
    plt.savefig(f"{name}.png")

if __name__=="__main__":
    typer.run(main)
