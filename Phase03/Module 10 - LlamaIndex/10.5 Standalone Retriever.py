# 10.5 Retriever - Used Independently (Halaman 119-120)
import os

from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()


def create_retriever() -> VectorIndexRetriever:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY belum tersedia. Tambahkan key tersebut ke file .env."
        )

    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=0.1,
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )

    documents = [
        Document(
            text=(
                "ChromaDB is an open-source embedding database with a simple "
                "Python API."
            ),
            metadata={"tag": "vector-db"},
        ),
        Document(
            text=(
                "Pinecone is a managed vector database designed for production "
                "workloads."
            ),
            metadata={"tag": "vector-db"},
        ),
        Document(
            text="FAISS is a library by Meta for efficient similarity search on dense vectors.",
            metadata={"tag": "vector-search"},
        ),
        Document(
            text="pgvector adds vector similarity search to PostgreSQL.",
            metadata={"tag": "vector-db"},
        ),
        Document(
            text="Weaviate is an open-source vector database with a GraphQL API.",
            metadata={"tag": "vector-db"},
        ),
    ]

    index = VectorStoreIndex.from_documents(documents)
    return VectorIndexRetriever(index=index, similarity_top_k=3)


if __name__ == "__main__":
    retriever = create_retriever()
    nodes = retriever.retrieve("open source vector databases")

    print("Retrieved nodes:")
    for node in nodes:
        score = node.score if node.score is not None else 0.0
        print(f" [{score:.4f}] {node.text}")
        print(f"  tag: {node.metadata.get('tag', 'unknown')}")
