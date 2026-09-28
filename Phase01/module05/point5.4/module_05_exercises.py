# 5.4 Module 05 Exercises
import numpy as np

def normalise_embeddings(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return matrix / norms

def pairwise_cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    normed = normalise_embeddings(vecs)
    return normed @ normed.T

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((5, 8))
    normed = normalise_embeddings(mat)
    print("Norms are 1.0:", np.allclose(np.linalg.norm(normed, axis=1), 1.0))
    print("Matrix shape:", pairwise_cosine_matrix(mat).shape)
