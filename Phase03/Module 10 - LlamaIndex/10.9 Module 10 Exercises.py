# 10.9 Module 10 Exercises (Halaman 124-125)
"""Q&A dokumen, perbandingan index, metadata filtering, dan reranking."""

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_index.core import (
    Document,
    Settings,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()


def configure_llama_index() -> None:
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


def load_txt_documents(folder: str | Path) -> list[Document]:
    """Membaca semua file .txt dari folder dan menyimpan nama file sebagai metadata."""
    directory = Path(folder)
    if not directory.is_dir():
        raise FileNotFoundError(f"Folder dokumen tidak ditemukan: {directory}")

    documents: list[Document] = []
    for file_path in sorted(directory.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(
                Document(
                    text=text,
                    metadata={"source": file_path.name},
                )
            )

    if not documents:
        raise ValueError(f"Tidak ada file .txt berisi teks di folder: {directory}")
    return documents


def build_qa_index(folder: str | Path) -> Any:
    documents = load_txt_documents(folder)
    return VectorStoreIndex.from_documents(documents)


_qa_query_engine: Any | None = None


def initialize_qa(folder: str | Path) -> None:
    global _qa_query_engine
    _qa_query_engine = build_qa_index(folder).as_query_engine()


def _answer_with_sources(query_engine: Any, question: str) -> dict[str, Any]:
    response = query_engine.query(question)
    sources = sorted(
        {
            str(node.metadata.get("source", "unknown"))
            for node in response.source_nodes
        }
    )
    return {"answer": str(response), "sources": sources}


def ask(question: str) -> dict[str, Any]:
    """Menjawab pertanyaan menggunakan folder yang telah diinisialisasi."""
    if _qa_query_engine is None:
        raise RuntimeError(
            "Q&A belum siap. Jalankan initialize_qa(folder) terlebih dahulu."
        )
    return _answer_with_sources(_qa_query_engine, question)


def compare_indexes(documents: list[Document], questions: list[str]) -> list[dict]:
    """Membandingkan VectorStoreIndex dan SummaryIndex pada pertanyaan yang sama."""
    if len(questions) != 5:
        raise ValueError("Perbandingan index harus menggunakan tepat 5 pertanyaan")

    vector_engine = VectorStoreIndex.from_documents(documents).as_query_engine(
        similarity_top_k=3
    )
    summary_engine = SummaryIndex.from_documents(documents).as_query_engine()

    results = []
    for question in questions:
        results.append(
            {
                "question": question,
                "vector_store": _answer_with_sources(vector_engine, question),
                "summary_index": _answer_with_sources(summary_engine, question),
            }
        )
    return results


def retrieve_by_metadata(
    index: VectorStoreIndex,
    question: str,
    topic: str,
    year: int,
    top_k: int = 3,
) -> list[NodeWithScore]:
    """Mengambil node yang cocok dengan topic DAN year."""
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="topic", value=topic),
            MetadataFilter(key="year", value=year),
        ]
    )
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
        filters=filters,
    )
    return retriever.retrieve(question)


def _parse_score(response: str) -> float:
    match = re.search(r"\b(?:10|[0-9])(?:\.[0-9]+)?\b", response)
    if not match:
        raise ValueError(f"Skor LLM tidak valid: {response!r}")
    return min(10.0, max(0.0, float(match.group())))


def rerank_top_nodes(
    retriever: VectorIndexRetriever,
    question: str,
    top_k: int = 10,
    final_k: int = 3,
) -> list[NodeWithScore]:
    """Retrieve top-10, nilai relevansinya dengan LLM, lalu kembalikan top-3."""
    nodes = VectorIndexRetriever(
        index=retriever._index,
        similarity_top_k=top_k,
        filters=retriever._filters,
    ).retrieve(question)

    scoring_prompt = (
        "Nilai relevansi teks berikut terhadap pertanyaan pada skala 0 sampai 10. "
        "Balas hanya angka.\n\nPertanyaan: {question}\nTeks: {text}"
    )
    scored_nodes = []
    for node in nodes:
        prompt = scoring_prompt.format(question=question, text=node.node.text)
        score_text = Settings.llm.complete(prompt)
        scored_nodes.append((node, _parse_score(str(score_text))))

    scored_nodes.sort(key=lambda item: item[1], reverse=True)
    return [
        NodeWithScore(node=node.node, score=score)
        for node, score in scored_nodes[:final_k]
    ]


if __name__ == "__main__":
    configure_llama_index()
    base_dir = Path(__file__).resolve().parent
    document_folder = base_dir / "documents"

    if document_folder.is_dir():
        initialize_qa(document_folder)
        print("Q&A:", ask("What information is in these documents?"))
    else:
        print(f"Buat folder .txt terlebih dahulu di: {document_folder}")

    demo_documents = [
        Document(
            text=(
                "RAG retrieves external context before generating an answer. "
                "It is useful when knowledge changes frequently."
            ),
            metadata={"source": "paragraph_01.txt", "year": 2024, "topic": "rag"},
        ),
        Document(
            text=(
                "Embeddings represent text as vectors so that semantically "
                "similar content can be searched."
            ),
            metadata={
                "source": "paragraph_02.txt",
                "year": 2024,
                "topic": "embeddings",
            },
        ),
        Document(
            text=(
                "A vector database stores embeddings and supports similarity "
                "search over those vectors."
            ),
            metadata={
                "source": "paragraph_03.txt",
                "year": 2024,
                "topic": "embeddings",
            },
        ),
        Document(
            text="Fine-tuning updates model weights using task-specific examples.",
            metadata={"source": "paragraph_04.txt", "year": 2023, "topic": "llm"},
        ),
        Document(
            text="Prompt engineering improves model responses by refining instructions.",
            metadata={"source": "paragraph_05.txt", "year": 2024, "topic": "prompting"},
        ),
        Document(
            text="Chunking divides long documents into smaller searchable sections.",
            metadata={"source": "paragraph_06.txt", "year": 2024, "topic": "rag"},
        ),
        Document(
            text="Metadata adds structured information such as date and subject.",
            metadata={"source": "paragraph_07.txt", "year": 2024, "topic": "retrieval"},
        ),
        Document(
            text="Reranking improves retrieval by applying a second relevance model.",
            metadata={"source": "paragraph_08.txt", "year": 2024, "topic": "retrieval"},
        ),
        Document(
            text="Evaluation measures whether an answer is accurate and grounded.",
            metadata={"source": "paragraph_09.txt", "year": 2024, "topic": "evaluation"},
        ),
        Document(
            text="LlamaIndex provides indexes, retrievers, and query engines for data.",
            metadata={"source": "paragraph_10.txt", "year": 2024, "topic": "llamaindex"},
        ),
    ]

    questions = [
        "What is RAG?",
        "How do embeddings represent text?",
        "What does a vector database store?",
        "Why is chunking useful?",
        "What is reranking?",
    ]
    print("\nPerbandingan VectorStoreIndex dan SummaryIndex:")
    for result in compare_indexes(demo_documents, questions):
        print(result)

    demo_index = VectorStoreIndex.from_documents(demo_documents)
    filtered = retrieve_by_metadata(
        demo_index,
        "How are embeddings used for search?",
        topic="embeddings",
        year=2024,
    )
    print("\nHasil metadata filter:")
    for node in filtered:
        print(node.node.metadata, node.node.text)

    demo_retriever = VectorIndexRetriever(index=demo_index, similarity_top_k=10)
    reranked = rerank_top_nodes(
        demo_retriever,
        "Which technique improves the relevance of retrieved results?",
    )
    print("\nTop-3 setelah reranking:")
    for node in reranked:
        print(f"[{node.score:.1f}] {node.node.text}")
