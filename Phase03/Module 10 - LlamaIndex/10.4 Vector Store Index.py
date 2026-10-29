import os

from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

def create_query_engine():
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
        # OpenAIEmbedding only accepts the unprefixed model name.
        model="text-embedding-3-small",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )

    documents = [
        Document(
            text=(
                "RAG combines retrieval with generation to answer questions "
                "from external knowledge."
            )
        ),
        Document(
            text=(
                "Fine-tuning trains a model on new data to improve performance "
                "on a specific task."
            )
        ),
        Document(
            text=(
                "Prompt engineering optimises the input to an LLM to get "
                "better outputs without any training."
            )
        ),
        Document(
            text=(
                "Vector databases store high-dimensional embeddings and support "
                "fast approximate nearest-neighbour search."
            )
        ),
        Document(
            text=(
                "LangChain provides abstractions for building chains, agents, "
                "and memory-enabled LLM applications."
            )
        ),
    ]

    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine(similarity_top_k=3)

if __name__ == "__main__":
    query_engine = create_query_engine()
    response = query_engine.query("What is the difference between RAG and fine-tuning?")
    print("Answer:", response)
    print("\nSource nodes:")
    for node in response.source_nodes:
        score = node.score if node.score is not None else 0.0
        print(f" Score {score:.4f}: {node.text[:60]}...")
