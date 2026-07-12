import os
import torch
import numpy as np
import matplotlib.pyplot as plt

LLAMA1_DIR = "results/llama1b"
LLAMA3_DIR = "results/llama3b"


def center(K):
    n = K.shape[0]
    H = torch.eye(n) - torch.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X, Y):
    """
    X: (num_examples, hidden_dim1)
    Y: (num_examples, hidden_dim2)
    """

    X = X.float()
    Y = Y.float()

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    X = X / (X.std(dim=0, keepdim=True) + 1e-8)
    Y = Y / (Y.std(dim=0, keepdim=True) + 1e-8)

    K = X @ X.T
    L = Y @ Y.T

    K = center(K)
    L = center(L)

    hsic = (K * L).sum()

    norm_x = torch.sqrt((K * K).sum())
    norm_y = torch.sqrt((L * L).sum())

    return (hsic / (norm_x * norm_y)).item()


llama1_layers = sorted(
    [f for f in os.listdir(LLAMA1_DIR) if f.endswith(".pt")],
    key=lambda x: int(x.split("_")[1].split(".")[0]),
)

llama3_layers = sorted(
    [f for f in os.listdir(LLAMA3_DIR) if f.endswith(".pt")],
    key=lambda x: int(x.split("_")[1].split(".")[0]),
)

similarity = np.zeros((len(llama1_layers), len(llama3_layers)))

for i, llama1_file in enumerate(llama1_layers):

    X = torch.load(os.path.join(LLAMA1_DIR, llama1_file))

    for j, llama3_file in enumerate(llama3_layers):

        Y = torch.load(os.path.join(LLAMA3_DIR, llama3_file))

        score = linear_cka(X, Y)

        similarity[i, j] = score

        print(
            f"{llama1_file} vs {llama3_file}: {score:.4f}"
        )

os.makedirs("results", exist_ok=True)
np.save(
    "results/cka_matrix_llama.npy",
    similarity
)

np.savetxt(
    "results/cka_matrix_llama.csv",
    similarity,
    delimiter=",",
    fmt="%.6f"
)

print("\nSaved CKA matrix.")