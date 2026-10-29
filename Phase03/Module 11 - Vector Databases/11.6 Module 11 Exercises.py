# 11.6 Module 11 Exercises (Halaman 133-134)
"""Benchmark dan utilitas production-style untuk ChromaDB."""

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import chromadb


EMBEDDING_DIMENSION = 32


def mock_embedding(text: str) -> list[float]:
    """Buat embedding deterministik tanpa API eksternal untuk kebutuhan latihan."""
    values: list[float] = []
    for index in range(EMBEDDING_DIMENSION):
        digest = hashlib.sha256(f"{index}:{text}".encode("utf-8")).digest()
        values.append((int.from_bytes(digest[:4], "big") / 2**31) - 1.0)
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def mock_embeddings(texts: list[str]) -> list[list[float]]:
    return [mock_embedding(text) for text in texts]


class VectorDBBenchmark:
    """Mengukur latency dan throughput query ChromaDB."""

    def __init__(self, client: Any, collection_name: str = "benchmark_documents"):
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def insert_mock_documents(self, count: int = 1000) -> None:
        documents = [
            f"Mock document {index}: information about retrieval and vector search."
            for index in range(count)
        ]
        self.collection.upsert(
            ids=[f"benchmark_{index}" for index in range(count)],
            documents=documents,
            embeddings=mock_embeddings(documents),
            metadatas=[{"index": index} for index in range(count)],
        )

    def run(self, query_count: int = 100, top_k: int = 5) -> dict[str, float]:
        if self.collection.count() == 0:
            raise ValueError("Masukkan dokumen benchmark sebelum menjalankan query.")

        latencies: list[float] = []
        query_embedding = mock_embedding("retrieval and vector search")
        for _ in range(query_count):
            started = time.perf_counter()
            self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "distances"],
            )
            latencies.append(time.perf_counter() - started)

        latencies.sort()
        percentile_index = min(
            len(latencies) - 1,
            max(0, math.ceil(len(latencies) * 0.95) - 1),
        )
        total_time = sum(latencies)
        return {
            "documents": float(self.collection.count()),
            "queries": float(query_count),
            "average_latency_ms": (total_time / query_count) * 1000,
            "p95_latency_ms": latencies[percentile_index] * 1000,
            "throughput_queries_per_second": query_count / total_time,
        }


class MultiCollectionSearch:
    """Mencari pada beberapa collection dan menggabungkan ranking-nya."""

    def __init__(self, client: Any):
        self.client = client

    def search(
        self,
        collection_names: list[str],
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not collection_names:
            raise ValueError("collection_names tidak boleh kosong.")

        merged: list[dict[str, Any]] = []
        for collection_name in collection_names:
            collection = self.client.get_collection(collection_name)
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "distances", "metadatas"],
            )
            documents = result["documents"][0] or []
            distances = result["distances"][0] or []
            metadatas = result["metadatas"][0] or []
            for document, distance, metadata in zip(
                documents, distances, metadatas
            ):
                merged.append(
                    {
                        "text": document,
                        "score": 1.0 - distance,
                        "distance": distance,
                        "collection": collection_name,
                        "metadata": metadata or {},
                    }
                )

        merged.sort(key=lambda item: item["score"], reverse=True)
        return merged[:top_k]


class TenantVectorStore:
    """Memisahkan data setiap tenant ke collection ChromaDB masing-masing."""

    def __init__(self, client: Any):
        self.client = client

    @staticmethod
    def _collection_name(tenant_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", tenant_id):
            raise ValueError("tenant_id hanya boleh berisi huruf, angka, '_' atau '-'.")
        return f"tenant_{tenant_id}"

    def _collection(self, tenant_id: str) -> Any:
        return self.client.get_or_create_collection(
            name=self._collection_name(tenant_id),
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        tenant_id: str,
        document_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        upsert_records: dict[str, Any] = {
            "ids": [document_id],
            "documents": [text],
            "embeddings": [mock_embedding(text)],
        }
        if metadata:
            upsert_records["metadatas"] = [metadata]
        self._collection(tenant_id).upsert(**upsert_records)

    def search(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        result = self._collection(tenant_id).query(
            query_embeddings=[mock_embedding(query)],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )
        return [
            {
                "text": text,
                "score": 1.0 - distance,
                "metadata": metadata or {},
            }
            for text, distance, metadata in zip(
                result["documents"][0] or [],
                result["distances"][0] or [],
                result["metadatas"][0] or [],
            )
        ]


def export_collection(collection: Any, output_file: str | Path) -> None:
    """Ekspor collection menjadi list dict JSON lengkap dengan embedding."""
    exported = collection.get(
        include=["documents", "embeddings", "metadatas"],
    )
    documents = exported["documents"] or []
    embeddings = (
        exported["embeddings"]
        if exported["embeddings"] is not None
        else []
    )
    metadatas = exported["metadatas"] or []
    records = [
        {
            "id": record_id,
            "text": text,
            "embedding": embedding.tolist()
            if hasattr(embedding, "tolist")
            else embedding,
            "metadata": metadata or {},
        }
        for record_id, text, embedding, metadata in zip(
            exported["ids"],
            documents,
            embeddings,
            metadatas,
        )
    ]
    Path(output_file).write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def import_collection(
    client: Any,
    collection_name: str,
    input_file: str | Path,
) -> Any:
    """Impor file JSON hasil export ke collection ChromaDB."""
    records = json.loads(Path(input_file).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Format export harus berupa JSON list.")

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    upsert_records: dict[str, Any] = {
        "ids": [record["id"] for record in records],
        "documents": [record["text"] for record in records],
        "embeddings": [record["embedding"] for record in records],
    }
    metadata = [record.get("metadata") for record in records]
    if any(metadata):
        upsert_records["metadatas"] = [
            item if item else {"imported": True}
            for item in metadata
        ]
    collection.upsert(**upsert_records)
    return collection


if __name__ == "__main__":
    client = chromadb.EphemeralClient()

    benchmark = VectorDBBenchmark(client)
    benchmark.insert_mock_documents(count=1000)
    print("Benchmark:", benchmark.run(query_count=100))

    first = client.get_or_create_collection("knowledge_a")
    second = client.get_or_create_collection("knowledge_b")
    first.upsert(
        ids=["a1"],
        documents=["RAG retrieves external knowledge."],
        embeddings=[mock_embedding("RAG retrieves external knowledge.")],
        metadatas=[{"group": "a"}],
    )
    second.upsert(
        ids=["b1"],
        documents=["Vector databases search embedding vectors."],
        embeddings=[mock_embedding("Vector databases search embedding vectors.")],
        metadatas=[{"group": "b"}],
    )
    search = MultiCollectionSearch(client)
    print(
        "Multi-collection:",
        search.search(
            ["knowledge_a", "knowledge_b"],
            mock_embedding("external knowledge search"),
        ),
    )

    tenant_store = TenantVectorStore(client)
    tenant_store.add("tenant_a", "doc1", "Private tenant A document.")
    tenant_store.add("tenant_b", "doc1", "Private tenant B document.")
    print("Tenant A:", tenant_store.search("tenant_a", "private document"))

    export_file = Path(__file__).with_name("chroma_export.json")
    export_collection(first, export_file)
    imported = import_collection(client, "knowledge_imported", export_file)
    print("Imported records:", imported.count())
    export_file.unlink(missing_ok=True)
