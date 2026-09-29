import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY belum tersedia di file .env")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)


def embed(
    texts: list[str],
    model: str = "openai/text-embedding-3-small"
) -> np.ndarray:
    """Embed a list of texts. Returns array of shape (n, dim)."""

    # API accepts up to 2048 texts per call
    response = client.embeddings.create(
        input=texts,
        model=model
    )

    # Sort by index to guarantee order matches input
    vectors = sorted(
        response.data,
        key=lambda e: e.index
    )

    return np.array(
        [v.embedding for v in vectors],
        dtype=np.float32
    )


texts = [
    "Retrieval-Augmented Generation combines search with LLMs.",
    "RAG retrieves documents then generates an answer from them.",
    "The Eiffel Tower is in Paris.",
    "Python is a popular programming language.",
    "Fine-tuning trains a model on new data.",
]


embeddings = embed(texts)

print(
    f"Shape: {embeddings.shape}"
)

print(
    f"Norm of first vector: "
    f"{np.linalg.norm(embeddings[0]):.4f}"
)  # ≈ 1.0