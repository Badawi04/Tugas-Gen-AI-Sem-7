# 9.3 Prompt Templates (Halaman 103-104)
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
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

# Simple template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in {language}."),
    ("human", "{question}"),
])

msgs = prompt.format_messages(
    role="Python tutor",
    language="English",
    question="What is a generator?",
)
print("Formatted simple prompt:", msgs)

response = llm.invoke(msgs)
print("\nModel answer:")
print(response.content)

# Template with conversation history placeholder
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Today's context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

formatted = chat_prompt.format_messages(
    context="User is learning about RAG",
    history=[
        HumanMessage(content="What is retrieval?"),
    ],
    input="How is it different from fine-tuning?",
)
print(f"\nTotal messages in formatted prompt: {len(formatted)}")

final_response = llm.invoke(formatted)
print("\nConversation answer:")
print(final_response.content)
