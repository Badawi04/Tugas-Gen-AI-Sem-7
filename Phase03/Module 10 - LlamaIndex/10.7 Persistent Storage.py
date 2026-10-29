# 10.7 Persistent Storage (Halaman 122-123)
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

PERSIST_DIR = Path(__file__).resolve().parent / "llama_index_store"


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


def create_documents() -> list[Document]:
    return [
        Document(text="LlamaIndex persists indexes to disk as JSON files."),
        Document(
            text="Loading a persisted index avoids re-embedding on every run."
        ),
    ]


def build_or_load_index() -> VectorStoreIndex:
    if PERSIST_DIR.exists():
        storage_context = StorageContext.from_defaults(
            persist_dir=str(PERSIST_DIR)
        )
        return load_index_from_storage(storage_context)

    index = VectorStoreIndex.from_documents(create_documents())
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    print(f"Index baru disimpan di: {PERSIST_DIR}")
    return index


if __name__ == "__main__":
    configure_llama_index()
    index = build_or_load_index()
    query_engine = index.as_query_engine()
    response = query_engine.query("How does LlamaIndex handle persistence?")
    print("After reload:", response)
