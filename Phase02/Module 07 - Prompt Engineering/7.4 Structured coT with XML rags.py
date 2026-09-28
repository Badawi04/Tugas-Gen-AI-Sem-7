import os
import re
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

SYSTEM = """Solve problems using this exact format:
<thinking>
Step-by-step reasoning here.
</thinking>
<answer>
The final answer only, no reasoning.
</answer>"""

resp = client.chat.completions.create(
    model=MODEL,
    max_tokens=512,
    messages=[
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": "A RAG pipeline retrieves 5 documents, each 400 tokens. The query is 50 tokens. The model has a 4096 token limit for context. How many tokens remain for the response?",
        },
    ],
    extra_body={"reasoning": {"exclude": True}},
)

if not resp.choices or not resp.choices[0].message.content:
    raise RuntimeError("OpenRouter tidak mengembalikan teks terstruktur.")

text = resp.choices[0].message.content

# Extract sections
thinking = re.search(
    r"<thinking>(.*?)</thinking>",
    text,
    re.DOTALL
)

answer = re.search(
    r"<answer>(.*?)</answer>",
    text,
    re.DOTALL
)

print(
    "Reasoning:",
    thinking.group(1).strip() if thinking else "not found"
)

if not answer:
    answer_text = text.strip()
else:
    answer_text = answer.group(1).strip()

print(
    "Answer: ",
    answer_text
)