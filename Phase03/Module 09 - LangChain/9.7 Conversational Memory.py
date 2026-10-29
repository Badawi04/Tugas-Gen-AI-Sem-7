# 9.7 Conversational Memory (Halaman 110-112)
import os
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

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

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor. Be concise."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | parser

# Session store - keyed by session_id
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap chain with history management
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

if __name__ == "__main__":
    config = {"configurable": {"session_id": "user-123"}}
    # Turn 1
    r1 = chain_with_history.invoke({"input": "What is RAG?"}, config=config)
    print("Turn 1:", r1[:100])
    # Turn 2 - model remembers turn 1
    r2 = chain_with_history.invoke({"input": "What are the main components I just asked about?"}, config=config)
    print("Turn 2:", r2[:100])
    # Inspect stored history
    history = get_session_history("user-123")
    print(f"Messages in history: {len(history.messages)}")
