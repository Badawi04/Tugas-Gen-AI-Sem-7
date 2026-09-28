from openai import OpenAI
import os, numpy as np
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)


def embed(
    texts: list[str],
    model: str = "text-embedding-3-small"
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