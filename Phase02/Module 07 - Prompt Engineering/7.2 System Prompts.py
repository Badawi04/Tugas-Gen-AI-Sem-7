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

# Poor system prompt - vague, no constraints
WEAK_SYSTEM = "You are an AI assistant."

# Strong system prompt - explicit role, rules, format
STRONG_SYSTEM = """You are a senior Python engineer reviewing
code for a production AI pipeline.
Your job:
- Identify bugs, security issues, and performance problems
- Suggest concrete improvements with code examples
- Explain WHY each issue matters
Rules:
- Be direct. Do not pad with compliments.
- If code is correct, say so briefly and move on.
- Always include the corrected code when suggesting a fix.
Format:
Return your review as a numbered list. Each item: Issue → Impact → Fix.
Return plain text only. Do not emit tool calls or special markup."""

messages = [
    {
        "role": "user",
        "content": """Review this function:
def get_user(user_id):
key = os.getenv('DB_KEY')
result = requests.get(f'http://db/{user_id}?key={key}')
return result.json()"""
    }
]

response = client.chat.completions.create(
    model="openrouter/free",
    max_tokens=1024,
    messages=[
        {"role": "system", "content": STRONG_SYSTEM},
        *messages,
    ],
    extra_body={"reasoning": {"exclude": True}},
)

if not response.choices or not response.choices[0].message.content:
    raise RuntimeError("OpenRouter tidak mengembalikan teks review.")

print(response.choices[0].message.content)