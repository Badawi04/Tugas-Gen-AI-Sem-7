import numpy as np
import os
from pathlib import Path

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

result = client.embeddings.create(
    model="openai/text-embedding-3-small",
    input=[
        "What is RAG?",
        "Explain vector databases."
    ]
)

embeddings = np.array(
    [item.embedding for item in result.data],
    dtype=np.float32
)

print(
    f"Shape: {embeddings.shape}"
)  # (2, 1024)

print(
    f"Token usage: {result.usage.total_tokens}"
)