# 10.8 Custom Query Pipeline (Halaman 123-124)
import os

from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
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
                "HNSW (Hierarchical Navigable Small World) is the graph "
                "algorithm powering most modern vector search."
            )
        ),
        Document(
            text=(
                "IVF (Inverted File Index) partitions vectors into clusters "
                "for fast approximate search."
            )
        ),
        Document(
            text=(
                "PQ (Product Quantisation) compresses vectors to reduce "
                "memory usage at the cost of recall."
            )
        ),
        Document(
            text=(
                "Exact nearest-neighbour search guarantees correctness but "
                "is O(n), which is impractical at scale."
            )
        ),
    ]

    index = VectorStoreIndex.from_documents(documents)
    return VectorIndexRetriever(index=index, similarity_top_k=2)


def run_query_modes(retriever: VectorIndexRetriever, question: str) -> None:
    for mode in (ResponseMode.COMPACT, ResponseMode.TREE_SUMMARIZE):
        synthesizer = get_response_synthesizer(response_mode=mode)
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
        )
        response = query_engine.query(question)
        print(f"[{mode}] {str(response)[:120]}\n")


if __name__ == "__main__":
    retriever = create_retriever()
    run_query_modes(retriever, "What algorithms are used in vector search?")
