import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY belum tersedia di file .env")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)
model = "openrouter/free"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window and pricing for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model identifier."
                    }
                },
                "required": ["model_name"]
            }
        }
    }
]

def get_model_info(model_name: str) -> dict:
    db = {
        "gpt-4o": {
            "context_k": 128,
            "cost_input": 2.50
        },
        "claude-sonnet-4-5": {
            "context_k": 200,
            "cost_input": 3.00
        },
    }

    return db.get(model_name, {"error": "unknown model"})


messages = [
    {
        "role": "user",
        "content": "What is gpt-4o's context window?"
    }
]

response = client.chat.completions.create(
    model=model,
    tools=tools,
    messages=messages
)

assistant_message = response.choices[0].message

if response.choices[0].finish_reason == "tool_calls" and assistant_message.tool_calls:
    tool_call = assistant_message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    result = get_model_info(**args)

    # Append assistant message and tool result
    messages.append(assistant_message.model_dump(exclude_none=True))
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

    final = client.chat.completions.create(
        model=model,
        messages=messages
    )

    print(final.choices[0].message.content)
else:
    print(assistant_message.content or "Model tidak mengembalikan tool call.")