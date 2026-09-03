"""
services/llm.py — Groq LLM client (singleton).

Single public function:
    generate(context: str, query: str) -> str

Uses the Groq Python SDK. The client is lazily initialised on first call
and reused for all subsequent requests.

Configuration (from .env via config.py):
    GROQ_API_KEY   — your Groq API key
    LLM_MODEL      — model ID (default: llama-3.3-70b-versatile)

Raises:
    RuntimeError  — if GROQ_API_KEY is not set when generate() is called
"""

from __future__ import annotations

import logging

from config import settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a temporally-aware knowledge assistant.

Rules:
- Answer using ONLY information from the provided sources.
- Cite every claim with [SOURCE N] inline (e.g. "Autograd uses dynamic graphs [SOURCE 1].").
- Prefer information from sources marked "← PREFERRED" when conflicts exist.
- If "⚠ TEMPORAL CONFLICT DETECTED" appears in the context, acknowledge it in your answer.
- Always include the validity date in citations where available: "As of [date], ..."
- If ANSWER CONFIDENCE is LOW, explicitly tell the user that conflicting information \
exists and recommend verifying from primary sources.
- If the sources do not contain enough information to answer, say so clearly.
- Never invent facts, version numbers, or API names not present in the context.
"""

class ModelUnavailableError(RuntimeError):
    """
    Raised when LLM_MODEL names a model the account cannot use.

    Subclasses RuntimeError so routers/query.py's existing handling (surface
    the message in the answer field rather than failing the whole request)
    applies — retrieval succeeded and its results are still worth returning.
    """


# Lazy singleton — created on first call to generate()
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file and restart the server."
        )

    from groq import Groq
    _client = Groq(api_key=settings.groq_api_key)
    log.info("Groq client initialised (model=%s).", settings.llm_model)
    return _client


def generate(context: str, query: str) -> str:
    """
    Call Groq to generate an answer grounded in the provided context.

    Args:
        context: Formatted context string from context.build_context().
        query:   The user's original question.

    Returns:
        The model's answer as a plain string (may contain [SOURCE N] citations).

    Raises:
        RuntimeError: If GROQ_API_KEY is not configured.
    """
    client = _get_client()

    user_message = f"Sources:\n\n{context}\n\nQuestion: {query}"

    log.debug("Calling Groq model=%s, context_len=%d chars", settings.llm_model, len(context))

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.1,    # low temperature for factual accuracy
            max_tokens=1024,
        )
    except Exception as exc:
        # A decommissioned model is the single most likely cause here and the
        # raw provider error ("model_not_found") does not tell the operator
        # what to do about it. Retrieval is unaffected, which makes this easy
        # to misdiagnose as a pipeline bug.
        if "model_not_found" in str(exc) or "does not exist" in str(exc):
            raise ModelUnavailableError(
                f"The configured LLM_MODEL {settings.llm_model!r} is not "
                f"available on your Groq account — hosted models are retired "
                f"periodically. Run `python scripts/list_llm_models.py` to see "
                f"what your key can reach, then update LLM_MODEL in .env."
            ) from exc
        raise

    answer = response.choices[0].message.content or ""
    log.info(
        "Groq answered: %d chars, finish_reason=%s",
        len(answer),
        response.choices[0].finish_reason,
    )
    return answer
