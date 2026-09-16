# 4.3 Environment Variables and Secrets

# Never hardcode API keys. Use environment variables, loaded via
# python-dotenv in development.

# Create a .env file in your project root:
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...

# Add to .gitignore immediately:
# echo ".env" >> .gitignore

import os
from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ


def get_api_key(provider: str) -> str:
    """Retrieve an API key from the environment."""
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    env_var = key_map.get(provider.lower())
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")
    key = os.getenv(env_var)
    if not key:
        raise EnvironmentError(
            f"{env_var} is not set. Add it to your .env file."
        )
    return key


# In practice:
# anthropic_key = get_api_key("anthropic")
# openai_key = get_api_key("openai")