# 9.9 Module 09 Exercises
"""Contoh LCEL, retrieval multi-query, map-reduce, dan evaluasi RAG."""

import hashlib
import json
import os
import re
from collections.abc import Callable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY belum tersedia di file .env")

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
    temperature=0.7,
    api_key=api_key,
)
parser = StrOutputParser()


def _read_json_list(response: str, item_name: str) -> list[str]:
    """Ambil list JSON dari jawaban model, termasuk jika dibungkus markdown."""
    candidate = response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)

    try:
        values = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM tidak mengembalikan JSON {item_name} yang valid") from exc

    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Format JSON {item_name} harus berupa list of strings")
    return values


# ---------------------------------------------------------------------------
# 1. LCEL chain: dokumen mentah -> 3 fakta -> 3 pertanyaan kuis
# ---------------------------------------------------------------------------
extract_facts_prompt = ChatPromptTemplate.from_template(
    """Baca dokumen berikut dan ambil tepat 3 fakta terpenting.
Balas hanya dengan JSON array berisi tiga string fakta.

Dokumen:
{document}"""
)

create_quiz_prompt = ChatPromptTemplate.from_template(
    """Buat satu pertanyaan kuis untuk setiap fakta berikut.
Jumlah pertanyaan harus sama dengan jumlah fakta dan urutannya harus sama.
Balas hanya dengan JSON array berisi string pertanyaan.

Fakta:
{facts}"""
)


def _parse_facts(response: str) -> dict[str, list[str]]:
    facts = _read_json_list(response, "fakta")
    if len(facts) != 3:
        raise ValueError("Tahap ekstraksi harus menghasilkan tepat 3 fakta")
    return {"facts": facts}


def _parse_questions(response: str) -> list[str]:
    questions = _read_json_list(response, "pertanyaan")
    if len(questions) != 3:
        raise ValueError("Tahap kuis harus menghasilkan tepat 3 pertanyaan")
    return questions


quiz_chain = (
    extract_facts_prompt
    | llm
    | parser
    | RunnableLambda(_parse_facts)
    | (lambda values: {"facts": "\n".join(f"- {fact}" for fact in values["facts"])})
    | create_quiz_prompt
    | llm
    | parser
    | RunnableLambda(_parse_questions)
)


# ---------------------------------------------------------------------------
# 2. MultiQueryRetriever manual
# ---------------------------------------------------------------------------
def multi_query_retrieve(
    query: str,
    retriever,
    paraphrase_fn: Callable[[str], list[str]],
) -> list[Document]:
    """Cari dengan query asli dan tiga parafrase, lalu hapus duplikasi isi."""
    paraphrases = paraphrase_fn(query)
    if len(paraphrases) != 3:
        raise ValueError("Fungsi parafrase harus mengembalikan tepat 3 query")

    unique_documents: list[Document] = []
    seen_hashes: set[str] = set()
    for candidate in [query, *paraphrases]:
        for document in retriever.invoke(candidate):
            content_hash = hashlib.sha256(
                document.page_content.encode("utf-8")
            ).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_documents.append(document)
    return unique_documents


def paraphrase_with_llm(query: str) -> list[str]:
    """Gunakan satu panggilan model untuk membuat tiga variasi query."""
    prompt = ChatPromptTemplate.from_template(
        "Tulis tepat 3 parafrase pertanyaan berikut dalam JSON array. "
        "Pertahankan maksudnya dan balas hanya JSON.\nPertanyaan: {query}"
    )
    response = (prompt | llm | parser).invoke({"query": query})
    return _read_json_list(response, "parafrase")


# ---------------------------------------------------------------------------
# 3. Map-reduce summarizer untuk dokumen yang melebihi context window
# ---------------------------------------------------------------------------
def map_reduce_summarize(
    document_text: str,
    summarize_chunk: Callable[[str], str],
    combine_summaries: Callable[[str], str],
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
) -> str:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(document_text)
    if not chunks:
        raise ValueError("Dokumen yang akan diringkas tidak boleh kosong")

    partial_summaries = [summarize_chunk(chunk) for chunk in chunks]
    return combine_summaries("\n".join(partial_summaries))


summarize_chunk_chain = (
    ChatPromptTemplate.from_template(
        "Ringkas bagian dokumen ini dalam satu atau dua kalimat:\n{text}"
    )
    | llm
    | parser
)
combine_summaries_chain = (
    ChatPromptTemplate.from_template(
        "Gabungkan ringkasan berikut menjadi satu ringkasan utuh:\n{text}"
    )
    | llm
    | parser
)


# ---------------------------------------------------------------------------
# 4. RAG dengan penilaian groundedness setelah jawaban dibuat
# ---------------------------------------------------------------------------
rag_answer_prompt = ChatPromptTemplate.from_template(
    """Jawab pertanyaan hanya berdasarkan konteks berikut.
Jika jawabannya tidak tersedia, katakan bahwa informasi tidak ditemukan.

Konteks:
{context}

Pertanyaan:
{question}"""
)

groundedness_prompt = ChatPromptTemplate.from_template(
    """Nilai seberapa kuat jawaban berikut didukung oleh konteks.
Gunakan skala 0 sampai 10 dan balas hanya satu angka.

Konteks:
{context}

Jawaban:
{answer}"""
)


def _score_number(response: str) -> float:
    match = re.search(r"\b(?:10|[0-9])(?:\.[0-9]+)?\b", response)
    if not match:
        raise ValueError("LLM penilai tidak mengembalikan skor 0-10")
    return min(10.0, max(0.0, float(match.group())))


def build_quality_checked_rag_chain(retriever):
    answer_chain = rag_answer_prompt | llm | parser
    judge_chain = groundedness_prompt | llm | parser | RunnableLambda(_score_number)

    def answer_and_score(question: str) -> dict:
        retrieved_documents = retriever.invoke(question)
        context = "\n\n".join(
            document.page_content for document in retrieved_documents
        )
        answer = answer_chain.invoke({"context": context, "question": question})
        score = judge_chain.invoke({"context": context, "answer": answer})
        return {"answer": answer, "score": score, "context": context}

    return RunnableLambda(answer_and_score)


if __name__ == "__main__":
    print("=== LATIHAN LANGCHAIN MODUL 09 ===\n")

    document = (
        "Retrieval-Augmented Generation menggabungkan model bahasa dengan "
        "dokumen eksternal. Retrieval mencari sumber yang relevan, sedangkan "
        "generation menyusun jawaban berdasarkan sumber tersebut. Pendekatan "
        "ini membantu mengurangi jawaban yang tidak didukung fakta."
    )

    print("1. LCEL Quiz Chain")
    print("   Pertanyaan:", quiz_chain.invoke(document))

    class MockRetriever:
        def invoke(self, query: str) -> list[Document]:
            return [
                Document(page_content="RAG menggunakan retrieval dan generation."),
                Document(page_content=f"Hasil pencarian untuk: {query}"),
            ]

    print("\n2. MultiQueryRetriever manual")
    demo_paraphrases = lambda query: [
        f"Apa pengertian {query}?",
        f"Bagaimana cara kerja {query}?",
        f"Apa manfaat utama {query}?",
    ]
    retrieved = multi_query_retrieve(
        "Retrieval-Augmented Generation",
        MockRetriever(),
        demo_paraphrases,
    )
    for item in retrieved:
        print("   -", item.page_content)

    print("\n3. Map-reduce summarizer")
    summary = map_reduce_summarize(
        document,
        lambda text: summarize_chunk_chain.invoke({"text": text}),
        lambda text: combine_summaries_chain.invoke({"text": text}),
    )
    print("   Ringkasan:", summary)

    print("\n4. RAG dengan quality filter")
    checked_rag = build_quality_checked_rag_chain(MockRetriever())
    print("   Hasil:", checked_rag.invoke("Apa itu RAG?"))
