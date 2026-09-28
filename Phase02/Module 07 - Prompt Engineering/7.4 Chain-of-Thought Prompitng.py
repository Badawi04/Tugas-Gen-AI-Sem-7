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
MODEL = "inclusionai/ling-3.0-flash-fin:free"

# Without CoT - model jumps to answer, more likely to be wrong
DIRECT_PROMPT = "If a model costs $3.00 per million input tokens and $15.00 per million output tokens, and a request uses 2,400 input tokens and 800 output tokens, what is the total cost in USD?"

# With CoT - model reasons through each step
COT_PROMPT = """If a model costs $3.00 per million input tokens and $15.00 per million output tokens,
and a request uses 2,400 input tokens and 800 output tokens,
what is the total cost in USD?
Think through this step by step before giving the final answer."""

# Zero-shot CoT: just adding "think step by step"
ZERO_SHOT_COT = """Solve this problem. Think step by step, showing each calculation.
Finally, state: ANSWER: $X.XXXXXX
Problem: A pipeline makes 50 API calls per hour. Each call uses an average of 1,200 input tokens
and 400 output tokens. The model costs $3.00/M input and $15.00/M output.
What is the daily cost?"""

for label, prompt in [
    ("Direct", DIRECT_PROMPT),
    ("CoT", COT_PROMPT),
    ("Zero-shot CoT", ZERO_SHOT_COT)
]:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    if not resp.choices:
        raise RuntimeError("OpenRouter tidak mengembalikan pilihan jawaban.")

    message = resp.choices[0].message
    answer = message.content or getattr(message, "reasoning", "")
    if not answer:
        raise RuntimeError("OpenRouter tidak mengembalikan teks jawaban.")

    print(f"==={label} ===")
    print(answer[:300])
    print()