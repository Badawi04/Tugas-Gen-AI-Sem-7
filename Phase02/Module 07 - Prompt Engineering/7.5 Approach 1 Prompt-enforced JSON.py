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
MODEL = "openrouter/free"

SYSTEM = """You are a data extractor. Extract information and
return ONLY a JSON object.
No markdown, no explanation, no code fences. Raw JSON only.
Schema:
{
"company": string,
"founded": integer or null,
"products": [string],
"headquarters": string or null,
"is_public": boolean
}"""

texts = [
    "Anthropic was founded in 2021 by Dario Amodei and others. It makes Claude AI models and is headquartered in San Francisco. It is a private company.",
    "OpenAI, founded in 2015, created ChatGPT and GPT-4. Based in San Francisco, it remains private despite a major Microsoft investment.",
]

def extract_company_info(text: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
    )

    if not resp.choices:
        raise RuntimeError("OpenRouter tidak mengembalikan teks JSON.")

    message = resp.choices[0].message
    raw = (message.content or getattr(message, "reasoning", "")).strip()
    if not raw:
        raise RuntimeError("OpenRouter tidak mengembalikan teks JSON.")

    # Strip any accidental markdown fences
    raw = (
        raw.removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

    return json.loads(raw)


for text in texts:
    info = extract_company_info(text)
    print(json.dumps(info, indent=2))
    print()