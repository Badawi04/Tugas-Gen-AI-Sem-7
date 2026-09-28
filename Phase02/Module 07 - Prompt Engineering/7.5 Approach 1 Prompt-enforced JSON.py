import anthropic, os, json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

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
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=256,
        system=SYSTEM,
        messages=[{"role": "user", "content": text}],
    )

    raw = resp.content[0].text.strip()

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