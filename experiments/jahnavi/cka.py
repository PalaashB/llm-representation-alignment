import os
import torch
import numpy as np
import matplotlib.pyplot as plt

TINY_DIR = "results/tinyllama"
QWEN_DIR = "results/qwen"


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

    K = X @ X.T
    L = Y @ Y.T

    K = center(K)
    L = center(L)

    hsic = (K * L).sum()

    norm_x = torch.sqrt((K * K).sum())
    norm_y = torch.sqrt((L * L).sum())

    return (hsic / (norm_x * norm_y)).item()


tiny_layers = sorted(
    [f for f in os.listdir(TINY_DIR) if f.endswith(".pt")]
)

qwen_layers = sorted(
    [f for f in os.listdir(QWEN_DIR) if f.endswith(".pt")]
)

similarity = np.zeros((len(tiny_layers), len(qwen_layers)))

for i, tiny_file in enumerate(tiny_layers):

    X = torch.load(os.path.join(TINY_DIR, tiny_file))

    for j, qwen_file in enumerate(qwen_layers):

        Y = torch.load(os.path.join(QWEN_DIR, qwen_file))

        score = linear_cka(X, Y)

        similarity[i, j] = score

        print(
            f"{tiny_file} vs {qwen_file}: {score:.4f}"
        )

os.makedirs("results", exist_ok=True)

np.save("results/cka_matrix.npy", similarity)

print("\nSaved CKA matrix.")