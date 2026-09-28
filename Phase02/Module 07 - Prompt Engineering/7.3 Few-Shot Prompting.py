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

FEW_SHOT_SYSTEM = """You are a data extractor. Given a raw AI
benchmark result string,
extract: model name, task, and score as a JSON object.
Examples:
Input: "GPT-4o scored 87.3% on the MMLU science subset"
Output: {"model": "gpt-4o", "task": "MMLU science", "score":
87.3}
Input: "Claude Sonnet 4.5 achieved 92.1 on HumanEval"
Output: {"model": "claude-sonnet-4-5", "task": "HumanEval",
"score": 92.1}
Input: "Gemini 1.5 Pro: 78.9% accuracy on GSM8K math"
Output: {"model": "gemini-1.5-pro", "task": "GSM8K math", "score":
78.9}
Return ONLY the JSON object. No explanation."""

test_inputs = [
    "GPT-4o-mini reached 82.0% on MMLU",
    "Llama 3.1 70B: 86.4 on TruthfulQA",
    "Claude Opus 4.5 scored 96.7% on SWE-bench Verified",
]

for text in test_inputs:
    resp = client.chat.completions.create(
        model="anthropic/claude-sonnet-4.5",
        max_tokens=256,
        messages=[
            {"role": "system", "content": FEW_SHOT_SYSTEM},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        extra_body={"reasoning": {"exclude": True}},
    )

    content = resp.choices[0].message.content if resp.choices else None
    if not content:
        raise RuntimeError("OpenRouter tidak mengembalikan JSON.")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start < 0 or end <= start:
            raise RuntimeError(f"Respons bukan JSON valid: {content}")
        parsed = json.loads(content[start:end])

    print(f"Input:{text}")
    print(f"Output:{json.dumps(parsed, ensure_ascii=False)}\n")