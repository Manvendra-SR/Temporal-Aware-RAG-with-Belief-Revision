"""
services/context.py — Context string builder for the LLM prompt.

build_context(chunks, budget) → (context_str, n_sources_included)

Output format:
    [SOURCE 1 | PyTorch v2.2 Docs]
    <content of chunk 1>

    [SOURCE 2 | Attention Is All You Need]
    <content of chunk 2>

    …

Adds sources in RRF-rank order until the cumulative token count
would exceed `budget`. This prevents sending an oversized prompt to the LLM.
"""

from __future__ import annotations

import logging

import tiktoken

from services.retriever import CandidateChunk

log = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


def build_context(chunks: list[CandidateChunk], budget: int = 3000) -> tuple[str, int]:
    """
    Build a formatted context string from a ranked list of chunks.

    Args:
        chunks: Ordered list of CandidateChunk (best first).
        budget: Maximum total tokens to include in the context.

    Returns:
        (context_string, number_of_sources_included)
    """
    parts: list[str] = []
    used_tokens = 0
    included = 0

    for i, chunk in enumerate(chunks, start=1):
        header = f"[SOURCE {i} | {chunk.doc_title}]"
        block = f"{header}\n{chunk.content}"
        block_tokens = _count_tokens(block)

        if used_tokens + block_tokens > budget:
            log.debug(
                "Context budget reached at source %d (%d tokens used, budget=%d).",
                i, used_tokens, budget,
            )
            break

        parts.append(block)
        used_tokens += block_tokens
        included += 1

    context = "\n\n".join(parts)
    log.debug("build_context: %d sources, %d tokens", included, used_tokens)
    return context, included
