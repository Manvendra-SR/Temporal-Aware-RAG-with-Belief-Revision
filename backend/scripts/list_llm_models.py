"""
scripts/list_llm_models.py — List the Groq models your API key can reach.

Groq retires hosted models periodically. When LLM_MODEL names a model that has
been decommissioned, retrieval keeps working and only answer generation fails,
which is easy to misread as a bug in the pipeline. Run this to see the current
list and pick a valid value for LLM_MODEL in .env.

Usage (from the backend/ directory, with the virtualenv active):

    python scripts/list_llm_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402


def main() -> int:
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set in .env — nothing to list.")
        return 1

    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    models = sorted(m.id for m in client.models.list().data)

    print(f"Currently configured LLM_MODEL: {settings.llm_model}")
    if settings.llm_model in models:
        print("  ✓ reachable with this key\n")
    else:
        print("  ✗ NOT reachable with this key — pick one of the models below\n")

    print(f"{len(models)} models available:")
    for model_id in models:
        print(f"  {model_id}")

    print(
        "\nNote: this list includes speech and safety models. For answer "
        "generation choose a general chat model."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
