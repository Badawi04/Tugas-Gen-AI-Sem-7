import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

if not os.getenv("OPENROUTER_API_KEY"):
    raise RuntimeError("OPENROUTER_API_KEY belum tersedia di file .env")

MODEL = "openrouter/free"

# OpenRouter tidak menyediakan count_tokens universal untuk semua model.
# Estimasi ini tidak melakukan API call dan tidak menimbulkan biaya.
def estimate_tokens(text: str) -> int:
    """Estimate tokens using roughly four characters per token."""
    return max(1, (len(text) + 3) // 4)

system_prompt = "You are a concise assistant."
user_prompt = "Explain the transformer architecture."
input_tokens = estimate_tokens(f"{system_prompt}\n{user_prompt}")
print(f"Estimated input tokens: {input_tokens}")

# Cost estimator
PRICING = {
    # Harga dalam USD per 1 juta token. Router gratis tidak dikenakan biaya.
    "openrouter/free": {"input": 0.00, "output": 0.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD."""
    if model not in PRICING:
        raise ValueError(f"Unknown model: {model}")

    p = PRICING[model]

    return (
        input_tokens * p["input"] +
        output_tokens * p["output"]
    ) / 1_000_000


cost = estimate_cost(
    MODEL,
    input_tokens=500,
    output_tokens=300
)

print(f"Estimated cost: ${cost:.6f}")

# Context window limits (always check before sending long documents)
CONTEXT_LIMITS = {
    "openrouter/free": 128_000,
}


def fits_in_context(
    model: str,
    token_count: int,
    reserve_for_output: int = 2048
) -> bool:
    limit = CONTEXT_LIMITS.get(model, 128_000)
    return token_count + reserve_for_output <= limit