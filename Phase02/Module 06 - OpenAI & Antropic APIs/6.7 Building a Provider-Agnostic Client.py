from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ChatResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class BaseLLMClient(ABC):

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        ...


class OpenRouterClient(BaseLLMClient):

    def __init__(self, model: str = "openrouter/free"):
        self.model = model
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY belum tersedia di file .env")

        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def chat(
        self,
        messages,
        system="",
        max_tokens=1024,
        temperature=0.7
    ) -> ChatResponse:
        api_messages = []

        if system:
            api_messages.append({
                "role": "system",
                "content": system
            })

        api_messages += [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=api_messages,
        )
        if not resp.choices:
            raise RuntimeError("OpenRouter tidak mengembalikan respons.")

        usage = resp.usage

        return ChatResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
        )


if __name__ == "__main__":
    client: BaseLLMClient = OpenRouterClient()
    msgs = [ChatMessage(role="user", content="What is a vector database?")]
    result = client.chat(msgs, system="Be concise.")
    print(result.text)
    print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")
