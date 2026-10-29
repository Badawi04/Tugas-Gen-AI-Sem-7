from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY belum tersedia di file .env")

# OpenRouter model. Use a model that is still available on OpenRouter.
# "inclusionai/ling-3.0-flash-sante:free" was removed, so use a supported alias.
llm_oai = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
    temperature=0.7,
    max_tokens=1024,
    api_key=api_key,
)

# Same interface, valid model alias
llm_ant = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
    temperature=0.7,
    max_tokens=1024,
    api_key=api_key,
)

# Both accept the same message types
messages = [
    SystemMessage(content="You are a concise technical assistant."),
    HumanMessage(content="What is LangChain Expression Language?"),
]

if __name__ == "__main__":
    print("=== Invoke model ===")
    response = llm_ant.invoke(messages)
    print(response.content)

    print("\n=== Stream model ===")
    for chunk in llm_ant.stream(messages):
        print(chunk.content, end="", flush=True)
    print()

    print("\n=== Batch model ===")
    responses = llm_ant.batch([
        [HumanMessage(content="What is RAG?")],
        [HumanMessage(content="What is a vector database?")],
    ])
    for r in responses:
        print(r.content[:80])
