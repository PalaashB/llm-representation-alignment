import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# -----------------------------
# Load CKA matrix
# -----------------------------

matrix = np.load("results/cka_matrix_llama.npy")

os.makedirs("figures", exist_ok=True)

# -----------------------------
# Matplotlib Heatmap
# -----------------------------

plt.figure(figsize=(11, 9))

im = plt.imshow(
    matrix,
    origin="lower",
    aspect="auto",
    cmap="viridis",
)

plt.colorbar(im, label="Linear CKA Similarity")

plt.xlabel("Llama-3.2-3B Layer", fontsize=13)
plt.ylabel("Llama-3.2-1B Layer", fontsize=13)

plt.title(
    "Layer-wise Linear CKA Similarity\nLlama-3.2-1B vs Llama-3.2-3B",
    fontsize=16,
    fontweight="bold"
)

plt.xticks(
    range(matrix.shape[1]),
    [f"L{i}" for i in range(matrix.shape[1])],
    rotation=90
)

plt.yticks(
    range(matrix.shape[0]),
    [f"L{i}" for i in range(matrix.shape[0])]
)

plt.tight_layout()

# -----------------------------
# Save figures
# -----------------------------

plt.savefig(
    "figures/cka_heatmap_llama1b_llama3b.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "figures/cka_heatmap_llama1b_llama3b.pdf",
    bbox_inches="tight"
)

# -----------------------------
# Save interactive matplotlib figure
# -----------------------------

with open(
    "figures/cka_heatmap_llama1b_llama3b.fig.pickle",
    "wb",
) as f:
    pickle.dump(plt.gcf(), f)

# -----------------------------
# Interactive Plotly Heatmap
# -----------------------------

fig = px.imshow(
    matrix,
    color_continuous_scale="Viridis",
    labels={
        "x": "Llama-3.2-3B Layer",
        "y": "Llama-3.2-1B Layer",
        "color": "Linear CKA",
    },
    x=[f"L{i}" for i in range(matrix.shape[1])],
    y=[f"L{i}" for i in range(matrix.shape[0])],
    title="Layer-wise Linear CKA Similarity<br>Llama-3.2-1B vs Llama-3.2-3B",
)

fig.update_layout(
    width=900,
    height=800,
    font=dict(size=13),
)

fig.write_html(
    "figures/cka_heatmap_llama1b_llama3b.html"
)

print("Saved:")
print("  figures/cka_heatmap_llama1b_llama3b.png")
print("  figures/cka_heatmap_llama1b_llama3b.pdf")
print("  figures/cka_heatmap_llama1b_llama3b.fig.pickle")
print("  figures/cka_heatmap_llama1b_llama3b.html")

plt.show()