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

response = client.chat.completions.create(
    model="openrouter/free",
    response_format={"type": "json_object"},  # enforces valid JSON
    extra_body={"reasoning": {"exclude": True}},
    messages=[
        {
            "role": "system",
            "content": """Extract entities. Return JSON with
this schema:
{"people": [string], "organizations": [string], "locations":
[string]}"""
        },
        {
            "role": "user",
            "content": "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla."
        }
    ]
)

if not response.choices or not response.choices[0].message.content:
    raise RuntimeError("OpenRouter tidak mengembalikan JSON.")

raw = response.choices[0].message.content.strip()
result = json.loads(raw)

print(result)

# {'people': ['Elon Musk'], 'organizations': ['SpaceX', 'Tesla'], 'locations': ['Hawthorne, California']}
