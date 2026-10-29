# 11.3 Pinecone (Halaman 128-130)
# pip install pinecone openai python-dotenv
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1"
INDEX_NAME = "ai-knowledge-base"
EMBEDDING_DIMENSION = 1536


def create_embedding_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY belum tersedia. Tambahkan key tersebut ke file .env."
        )
    return OpenAI(api_key=api_key, base_url=OPENROUTER_URL)


def embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=texts,
    )
    return [
        item.embedding
        for item in sorted(response.data, key=lambda item: item.index)
    ]


def wait_until_ready(pc: Pinecone, index_name: str, timeout: int = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        description = pc.describe_index(index_name)
        status = description.status
        if status.get("ready", False):
            return
        time.sleep(2)
    raise TimeoutError(f"Index Pinecone belum siap setelah {timeout} detik.")


def run_demo() -> None:
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        print(
            "Pinecone dilewati karena PINECONE_API_KEY belum tersedia.\n"
            "Tambahkan key berikut ke file .env untuk menjalankan demo:\n"
            "PINECONE_API_KEY=your-pinecone-api-key"
        )
        return

    embedding_client = create_embedding_client()
    pinecone = Pinecone(api_key=pinecone_key)

    if INDEX_NAME not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        wait_until_ready(pinecone, INDEX_NAME)

    index = pinecone.Index(INDEX_NAME)
    documents = [
        {"id": "d1", "text": "RAG connects LLMs to external knowledge bases."},
        {"id": "d2", "text": "Fine-tuning updates model weights on task-specific data."},
        {"id": "d3", "text": "ChromaDB is an open-source local vector database."},
    ]

    vectors = [
        {
            "id": document["id"],
            "values": embed(embedding_client, [document["text"]])[0],
            "metadata": {"text": document["text"], "source": "guide"},
        }
        for document in documents
    ]
    index.upsert(vectors=vectors)
    print(index.describe_index_stats())

    query_vector = embed(
        embedding_client,
        ["How can external knowledge be added to an LLM?"],
    )[0]
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
    )

    print("\nTop results:")
    for match in results.matches:
        metadata = match.metadata or {}
        print(f"[{match.score:.4f}] {metadata.get('text', 'No text')}")


if __name__ == "__main__":
    run_demo()
