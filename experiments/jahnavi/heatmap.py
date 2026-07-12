import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# -----------------------------
# Load CKA matrix
# -----------------------------

matrix = np.load("results/cka_matrix.npy")

os.makedirs("figures", exist_ok=True)

# -----------------------------
# Matplotlib Heatmap
# -----------------------------

plt.figure(figsize=(10, 8))

im = plt.imshow(
    matrix,
    origin="lower",
    aspect="auto",
    cmap="viridis",
)

plt.colorbar(im, label="CKA Similarity")

plt.xlabel("Qwen Layer", fontsize=12)
plt.ylabel("TinyLlama Layer", fontsize=12)

plt.title(
    "Layer-wise CKA Similarity",
    fontsize=15,
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
# Save high-quality figures
# -----------------------------

plt.savefig(
    "figures/cka_heatmap.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "figures/cka_heatmap.pdf",
    bbox_inches="tight"
)

# -----------------------------
# Save interactive matplotlib figure
# -----------------------------

with open("figures/cka_heatmap.fig.pickle", "wb") as f:
    pickle.dump(plt.gcf(), f)

# -----------------------------
# Interactive Plotly Heatmap
# -----------------------------

fig = px.imshow(
    matrix,
    color_continuous_scale="Viridis",
    labels={
        "x": "Qwen Layer",
        "y": "TinyLlama Layer",
        "color": "CKA"
    },
    x=[f"L{i}" for i in range(matrix.shape[1])],
    y=[f"L{i}" for i in range(matrix.shape[0])],
    title="Layer-wise CKA Similarity"
)

fig.update_layout(
    width=850,
    height=750,
    font=dict(size=13)
)

fig.write_html("figures/cka_heatmap.html")

print("Saved:")
print("  figures/cka_heatmap.png")
print("  figures/cka_heatmap.pdf")
print("  figures/cka_heatmap.fig.pickle")
print("  figures/cka_heatmap.html")

plt.show()