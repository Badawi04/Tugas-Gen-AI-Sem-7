# 4.4 Module 04 Exercises

# ─────────────────────────────────────────────────────────────
# Exercise 1: Save and load conversation history as JSON
# ─────────────────────────────────────────────────────────────
import json
import threading
import time
from pathlib import Path


def save_conversation(history: list[dict], path: str) -> None:
    Path(path).write_text(json.dumps(history, indent=2), encoding="utf-8")


def load_conversation(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────
# Exercise 2: Async compare_endpoints using httpx
# ─────────────────────────────────────────────────────────────
import asyncio
import httpx


async def compare_endpoints(urls: list[str]) -> list[tuple[str, int, float]]:
    """Fetch all URLs concurrently and return (url, status_code, response_time_ms)."""
    async def fetch(client: httpx.AsyncClient, url: str) -> tuple[str, int, float]:
        start = time.perf_counter()
        try:
            response = await client.get(url, timeout=10.0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return (url, response.status_code, round(elapsed_ms, 2))
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return (url, -1, round(elapsed_ms, 2))

    async with httpx.AsyncClient() as client:
        tasks = [fetch(client, url) for url in urls]
        return list(await asyncio.gather(*tasks))


# ─────────────────────────────────────────────────────────────
# Exercise 3: Config loader with environment variable overrides
# ─────────────────────────────────────────────────────────────
import os


def load_config_with_env(config_path: str) -> dict:
    """Read JSON config and override with env vars (env takes precedence)."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    for key in config:
        env_value = os.getenv(key)
        if env_value is not None:
            config[key] = env_value
    return config


# ─────────────────────────────────────────────────────────────
# Exercise 4: Thread-safe CSV logger for LLM calls
# ─────────────────────────────────────────────────────────────
class ThreadSafeCSVLogger:
    def __init__(self, filename: str):
        self.filename = filename
        self.lock = threading.Lock()
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write("timestamp,model,input_tokens,output_tokens,latency_ms\n")

    def log(self, model: str, in_tok: int, out_tok: int, latency_ms: float):
        with self.lock:
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(f"{time.time()},{model},{in_tok},{out_tok},{latency_ms}\n")


# ─────────────────────────────────────────────────────────────
# Demo (opsional, jalankan untuk uji coba)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Demo Exercise 1
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    save_conversation(history, "conversation.json")
    loaded = load_conversation("conversation.json")
    print("Loaded conversation:", loaded)

    # Demo Exercise 2
    urls = [
        "https://jsonplaceholder.typicode.com/posts/1",
        "https://jsonplaceholder.typicode.com/posts/2",
    ]
    results = asyncio.run(compare_endpoints(urls))
    for url, status, ms in results:
        print(f"{url} -> {status} ({ms} ms)")

    # Demo Exercise 3
    # (butuh file config.json + env var; contoh di bawah hanya ilustrasi)
    # cfg = load_config_with_env("config.json")
    # print("Config:", cfg)

    # Demo Exercise 4
    logger = ThreadSafeCSVLogger("llm_calls.csv")
    logger.log("claude-sonnet-4-5", 350, 210, 420.5)
    logger.log("gpt-4o", 420, 180, 380.2)
    print("Logged 2 LLM calls to llm_calls.csv")