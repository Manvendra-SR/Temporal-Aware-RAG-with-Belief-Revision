"""
services/context.py — Context string builder for the LLM prompt.

build_context(chunks, budget, revision_result) → (context_str, n_sources_included)

Without revision_result (Phase 5 and below):
    [SOURCE 1 | PyTorch v2.2 Docs]
    <content of chunk 1>

    [SOURCE 2 | Attention Is All You Need]
    <content of chunk 2>

With revision_result (Phase 7+):
    [SOURCE 1 | pytorch_docs v2.2 | valid from 2024-01-15]   ← PREFERRED
    <content>

    ⚠ TEMPORAL CONFLICT DETECTED:
    pytorch_docs v1.13 (2022-12-15) and pytorch_docs v2.2 (2024-01-15) make conflicting claims.
    Source 1 is more recent and preferred. The older claim was:
    "torch.autograd.Variable should be used to wrap tensors"

    [SOURCE 2 | pytorch_docs v1.13 | valid from 2022-12-15]   [DEPRECATED]
    <content>

    ANSWER CONFIDENCE: medium
    REASON: Conflict resolved by temporal preference (>90 day gap)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import tiktoken

from services.retriever import CandidateChunk

if TYPE_CHECKING:
    from services.belief_revision import RevisionResult

log = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


def build_context(
    chunks: list[CandidateChunk],
    budget: int = 3000,
    revision_result: Optional["RevisionResult"] = None,
) -> tuple[str, int]:
    """
    Build a formatted context string from a ranked list of chunks.

    Args:
        chunks:          Ordered list of CandidateChunk (best first).
        budget:          Maximum total tokens to include in the context.
        revision_result: Optional RevisionResult from belief_revision.revise().
                         When provided, chunks are annotated with PREFERRED /
                         DEPRECATED markers and conflict notices are injected.

    Returns:
        (context_string, number_of_sources_included)
    """
    # Determine which chunk_ids are excluded (deprecated) per revision result
    exclude_set: set[str] = set()
    if revision_result is not None:
        exclude_set = set(revision_result.exclude_chunks)

    parts: list[str] = []
    used_tokens = 0
    included = 0

    # Inject conflict notices once at the top (before any source block)
    if revision_result is not None and revision_result.conflict_notices:
        notice_block = "\n\n".join(revision_result.conflict_notices)
        notice_tokens = _count_tokens(notice_block)
        if notice_tokens < budget:
            parts.append(notice_block)
            used_tokens += notice_tokens

    for i, chunk in enumerate(chunks, start=1):
        # Build header with temporal metadata when available
        header_parts = [f"SOURCE {i} | {chunk.doc_title}"]
        if chunk.version_string:
            header_parts.append(f"v{chunk.version_string}")
        if chunk.valid_from:
            header_parts.append(f"valid from {chunk.valid_from.strftime('%Y-%m-%d')}")

        header = "[" + " | ".join(header_parts) + "]"

        # Add PREFERRED / DEPRECATED annotation
        if revision_result is not None:
            if chunk.chunk_id in exclude_set:
                header += "   [DEPRECATED]"
            elif revision_result.include_chunks and chunk.chunk_id in revision_result.include_chunks:
                # Mark as PREFERRED only if it's specifically in include list
                # and there were actual conflicts processed
                if revision_result.belief_revision_applied:
                    header += "   ← PREFERRED"

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

    # Append confidence footer
    if revision_result is not None and revision_result.answer_confidence not in ("none", None):
        confidence_text = (
            f"ANSWER CONFIDENCE: {revision_result.answer_confidence.upper()}\n"
            f"REASON: {revision_result.confidence_reason}"
        )
        conf_tokens = _count_tokens(confidence_text)
        if used_tokens + conf_tokens <= budget + 200:  # small grace for footer
            parts.append(confidence_text)

    context = "\n\n".join(parts)
    log.debug("build_context: %d sources, %d tokens", included, used_tokens)
    return context, included
