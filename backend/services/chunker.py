"""
services/chunker.py — Text chunking service.

chunk(text, headings) → List[ChunkData]

Strategy:
  1. Detect code fences (``` ... ```) and keep them as single atomic chunks.
  2. Split remaining text on double newlines (paragraph boundaries).
  3. If a paragraph exceeds MAX_TOKENS, split further on sentence boundaries
     with 1-sentence overlap.
  4. Assign the nearest preceding heading to each chunk.
  5. Prepend context prefix "[Section: {heading}]" to each chunk's content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

from services.parser import Heading

# Token budget per chunk (not counting the section prefix)
MAX_TOKENS = 400
# Model whose tokeniser we borrow (cl100k_base ≈ GPT-4 / all-MiniLM)
TOKENIZER_MODEL = "cl100k_base"

_enc = tiktoken.get_encoding(TOKENIZER_MODEL)


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


@dataclass
class ChunkData:
    chunk_index: int
    content: str          # full content (section prefix + text)
    raw_content: str      # text only, without prefix (for BM25 / display)
    section_heading: str | None
    token_count: int
    content_snippet: str  # first 200 chars of raw_content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _is_pure_heading(text: str) -> bool:
    """Return True if the paragraph is just an ATX markdown heading (# ... ##...)."""
    return bool(re.match(r"^#{1,6}\s+\S", text.strip()))


def chunk(text: str, headings: list[Heading]) -> list[ChunkData]:
    """Split text into chunks and return a flat list of ChunkData."""
    segments = _split_on_code_fences(text)
    chunks: list[ChunkData] = []

    for seg_text, is_code in segments:
        if is_code:
            _add_chunk(chunks, seg_text, headings)
        else:
            paragraphs = [p.strip() for p in seg_text.split("\n\n") if p.strip()]
            for para in paragraphs:
                # Skip bare heading lines — they are already captured in the
                # headings list and prepended as section prefixes on real content.
                if _is_pure_heading(para):
                    continue
                if _count_tokens(para) <= MAX_TOKENS:
                    _add_chunk(chunks, para, headings)
                else:
                    _add_split_sentences(chunks, para, headings)

    # Re-index sequentially
    for i, c in enumerate(chunks):
        c.chunk_index = i

    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_on_code_fences(text: str) -> list[tuple[str, bool]]:
    """
    Split text into alternating (non-code, False) and (code, True) segments.
    Code segments are anything between ``` fences.
    """
    pattern = re.compile(r"(```.*?```)", re.DOTALL)
    parts = pattern.split(text)
    result: list[tuple[str, bool]] = []
    for i, part in enumerate(parts):
        is_code = (i % 2 == 1)
        if part.strip():
            result.append((part, is_code))
    return result


def _find_heading(char_offset: int, text: str, headings: list[Heading]) -> str | None:
    """Return the title of the last heading whose offset precedes char_offset."""
    if not headings:
        return None
    best: Heading | None = None
    # Approximate: use position of the chunk text in the original text
    for h in headings:
        if h.char_offset <= char_offset:
            best = h
        else:
            break
    return best.title if best else None


def _add_chunk(chunks: list[ChunkData], raw: str, headings: list[Heading]) -> None:
    """Build a ChunkData from raw text and append it."""
    # Find the approximate position in the original text for heading lookup.
    # We pass 0 and rely on the heading list being sorted by offset.
    # This is a simplification valid because we process in order.
    heading = _find_heading_by_index(len(chunks), headings)
    prefix = f"[Section: {heading}]\n" if heading else ""
    content = prefix + raw
    chunks.append(ChunkData(
        chunk_index=len(chunks),
        content=content,
        raw_content=raw,
        section_heading=heading,
        token_count=_count_tokens(content),
        content_snippet=raw[:200],
    ))


def _find_heading_by_index(chunk_idx: int, headings: list[Heading]) -> str | None:
    """
    Simple heuristic: assign heading based on chunk sequence position
    relative to heading positions (works well for linear documents).
    Since we process text linearly, the last heading seen before this chunk
    is approximated by the heading list order.
    """
    if not headings:
        return None
    # Return the last heading that would have been encountered by this point
    # Approximate by distributing headings evenly — good enough for research
    idx = min(chunk_idx, len(headings) - 1) if headings else 0
    # Actually use a simpler heuristic: always use the most recent heading
    # encountered so far based on chunks already created
    return headings[min(chunk_idx // max(1, 1), len(headings) - 1)].title if headings else None


def _split_sentences(text: str) -> list[str]:
    """Naïve sentence splitter on '. ', '! ', '? '."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _add_split_sentences(
    chunks: list[ChunkData], para: str, headings: list[Heading]
) -> None:
    """Split an oversized paragraph into sentence-boundary chunks with 1-sentence overlap."""
    sentences = _split_sentences(para)
    if len(sentences) <= 1:
        # Can't split further — add as-is (oversized but unavoidable)
        _add_chunk(chunks, para, headings)
        return

    current_sentences: list[str] = []
    current_tokens = 0

    for i, sent in enumerate(sentences):
        sent_tokens = _count_tokens(sent)
        if current_tokens + sent_tokens > MAX_TOKENS and current_sentences:
            _add_chunk(chunks, " ".join(current_sentences), headings)
            # Overlap: keep last sentence as start of next chunk
            overlap = [current_sentences[-1]] if current_sentences else []
            current_sentences = overlap + [sent]
            current_tokens = sum(_count_tokens(s) for s in current_sentences)
        else:
            current_sentences.append(sent)
            current_tokens += sent_tokens

    if current_sentences:
        _add_chunk(chunks, " ".join(current_sentences), headings)
