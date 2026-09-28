# pip install voyageai

import voyageai
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

vo = voyageai.Client(
    api_key=os.environ["VOYAGE_API_KEY"]
)

result = vo.embed(
    [
        "What is RAG?",
        "Explain vector databases."
    ],
    model="voyage-3",  # current recommended model
    input_type="document",  # "document" for corpus, "query" for search queries
)

embeddings = np.array(
    result.embeddings,
    dtype=np.float32
)

print(
    f"Shape: {embeddings.shape}"
)  # (2, 1024)

print(
    f"Token usage: {result.total_tokens}"
)