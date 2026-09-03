"""
services/chunker.py — Structure-aware, semantically refined text chunking.

Pipeline:
  Stage 1: Structural segmentation — split text on heading boundaries
  Stage 2: Block splitting         — classify content (prose/code/table/list)
  Stage 3: Semantic refinement     — split prose on sentence-embedding similarity
  Stage 4: Metadata attachment     — build ChunkData with offsets and section context

Public API:
    chunk(text, headings, embedder=None, config=None) → list[ChunkData]

When embedder is None, Stage 3 is skipped and prose is split on paragraph
boundaries + token budget only (equivalent to a simpler fallback mode).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tiktoken

from services.parser import Heading
from services.sentence_splitter import split_sentences

# ---------------------------------------------------------------------------
# Tokeniser (shared across the module)
# ---------------------------------------------------------------------------

TOKENIZER_MODEL = "cl100k_base"
_enc = tiktoken.get_encoding(TOKENIZER_MODEL)


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkingConfig:
    """All tuneable parameters for the chunking pipeline in one place."""
    max_tokens:         int   = 400    # hard upper limit per chunk
    min_chunk_tokens:   int   = 60     # minimum before a semantic split is allowed
    similarity_drop:    float = 0.65   # fraction of rolling mean that triggers split
    window_size:        int   = 3      # sentences in the similarity comparison window
    overlap_sentences:  int   = 1      # sentences of overlap between adjacent chunks
    min_section_tokens: int   = 30     # sections smaller than this merge into previous


DEFAULT_CONFIG = ChunkingConfig()


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChunkData:
    chunk_index:     int
    content:         str           # "[Section: {heading}]\n" + raw_content
    raw_content:     str           # text only (for BM25 / display)
    section_heading: str | None
    token_count:     int
    content_snippet: str           # first 200 chars of raw_content
    char_start:      int           # start offset in original text
    char_end:        int           # end offset in original text


# ---------------------------------------------------------------------------
# Internal dataclasses (not exposed outside this module)
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """A structural segment bounded by headings."""
    heading: str | None
    heading_level: int       # 1–6; 0 = no heading (preamble)
    text: str
    char_start: int
    char_end: int


@dataclass
class Block:
    """A typed content block within a section."""
    block_type: Literal["prose", "code", "table", "list"]
    text: str
    section: Section
    char_start: int
    char_end: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk(
    text: str,
    headings: list[Heading],
    embedder=None,
    config: ChunkingConfig | None = None,
) -> list[ChunkData]:
    """Split text into chunks and return a flat list of ChunkData."""
    cfg = config or DEFAULT_CONFIG
    if not text.strip():
        return []

    # Stage 1: structural segmentation
    sections = _segment(text, headings, cfg)

    # Stage 2 + 3: block splitting and refinement per section
    raw_chunks: list[tuple[str, Section, int, int]] = []  # (text, section, char_start, char_end)

    for section in sections:
        blocks = _split_blocks(section)
        for block in blocks:
            if block.block_type == "prose":
                prose_chunks = _refine_prose(block, embedder, cfg)
                raw_chunks.extend(prose_chunks)
            else:
                # Code, table, list blocks: keep atomic, or split on token budget
                _split_atomic(block, raw_chunks, cfg)

    # Stage 4: metadata attachment
    result: list[ChunkData] = []
    for i, (chunk_text, section, c_start, c_end) in enumerate(raw_chunks):
        heading = section.heading
        prefix = f"[Section: {heading}]\n" if heading else ""
        content = prefix + chunk_text
        result.append(ChunkData(
            chunk_index=i,
            content=content,
            raw_content=chunk_text,
            section_heading=heading,
            token_count=_count_tokens(content),
            content_snippet=chunk_text[:200],
            char_start=c_start,
            char_end=c_end,
        ))

    return result


# ---------------------------------------------------------------------------
# Stage 1: Structural Segmentation
# ---------------------------------------------------------------------------

def _segment(text: str, headings: list[Heading], cfg: ChunkingConfig) -> list[Section]:
    """Split text into sections bounded by headings."""
    if not headings:
        # No headings — entire document is one section
        return [Section(
            heading=None,
            heading_level=0,
            text=text,
            char_start=0,
            char_end=len(text),
        )]

    # Sort headings by char_offset (they should already be sorted, but be safe)
    sorted_headings = sorted(headings, key=lambda h: h.char_offset)

    sections: list[Section] = []

    # Preamble: text before the first heading
    first_offset = sorted_headings[0].char_offset
    if first_offset > 0:
        preamble = text[:first_offset].strip()
        if preamble:
            sections.append(Section(
                heading=None,
                heading_level=0,
                text=preamble,
                char_start=0,
                char_end=first_offset,
            ))

    # Each heading starts a section that extends until the next heading
    for i, h in enumerate(sorted_headings):
        start = h.char_offset
        end = sorted_headings[i + 1].char_offset if i + 1 < len(sorted_headings) else len(text)
        section_text = text[start:end]

        # Strip the heading line itself from the section content
        # (it will be used as a prefix, not embedded as content)
        heading_line_end = section_text.find("\n")
        if heading_line_end != -1:
            body = section_text[heading_line_end + 1:].strip()
            body_start = start + heading_line_end + 1
        else:
            body = ""
            body_start = start

        if body:
            sections.append(Section(
                heading=h.title,
                heading_level=h.level,
                text=body,
                char_start=body_start,
                char_end=end,
            ))

    # Merge tiny sections into the previous one
    merged: list[Section] = []
    for section in sections:
        tokens = _count_tokens(section.text)
        if tokens < cfg.min_section_tokens and merged and section.heading is None:
            # Only merge headingless fragments — real sections stay separate
            prev = merged[-1]
            merged[-1] = Section(
                heading=prev.heading,
                heading_level=prev.heading_level,
                text=prev.text + "\n\n" + section.text,
                char_start=prev.char_start,
                char_end=section.char_end,
            )
        else:
            merged.append(section)

    return merged


# ---------------------------------------------------------------------------
# Stage 2: Block Splitting
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"(```.*?```)", re.DOTALL)
_TABLE_LINE_RE = re.compile(r"^\|.*\|$")
_LIST_ITEM_RE = re.compile(r"^(?:[-*+]|\d+\.)\s")


def _split_blocks(section: Section) -> list[Block]:
    """Split a section into typed content blocks."""
    text = section.text
    base_offset = section.char_start

    # First pass: separate code fences
    parts = _CODE_FENCE_RE.split(text)
    blocks: list[Block] = []
    offset = 0

    for i, part in enumerate(parts):
        is_code = (i % 2 == 1)
        stripped = part.strip()
        if not stripped:
            offset += len(part)
            continue

        abs_start = base_offset + text.find(part, offset)
        abs_end = abs_start + len(part)

        if is_code:
            blocks.append(Block(
                block_type="code",
                text=stripped,
                section=section,
                char_start=abs_start,
                char_end=abs_end,
            ))
        else:
            # Second pass: split non-code on double newlines to get paragraphs
            _classify_paragraphs(stripped, section, abs_start, blocks)

        offset += len(part)

    return blocks


def _classify_paragraphs(
    text: str, section: Section, base_offset: int, blocks: list[Block]
) -> None:
    """Classify paragraphs into prose, table, or list blocks."""
    paragraphs = text.split("\n\n")
    offset = 0

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            offset += len(para) + 2  # +2 for \n\n
            continue

        # Skip bare ATX heading lines (already captured in headings list)
        if re.match(r"^#{1,6}\s+\S", stripped):
            offset += len(para) + 2
            continue

        abs_start = base_offset + text.find(para, offset)
        abs_end = abs_start + len(stripped)

        block_type = _detect_block_type(stripped)
        blocks.append(Block(
            block_type=block_type,
            text=stripped,
            section=section,
            char_start=abs_start,
            char_end=abs_end,
        ))
        offset += len(para) + 2


def _detect_block_type(text: str) -> Literal["prose", "code", "table", "list"]:
    """Classify a paragraph as prose, table, or list."""
    lines = text.strip().split("\n")

    # Table: ≥2 lines with pipes
    if len(lines) >= 2 and all(_TABLE_LINE_RE.match(l.strip()) for l in lines if l.strip()):
        return "table"

    # List: majority of lines start with list markers
    list_count = sum(1 for l in lines if _LIST_ITEM_RE.match(l.strip()))
    if list_count > 0 and list_count >= len(lines) * 0.6:
        return "list"

    return "prose"


# ---------------------------------------------------------------------------
# Stage 3: Semantic Refinement
# ---------------------------------------------------------------------------

def _refine_prose(
    block: Block,
    embedder,
    cfg: ChunkingConfig,
) -> list[tuple[str, Section, int, int]]:
    """
    Split a prose block into semantically coherent chunks.

    If embedder is None, falls back to paragraph-boundary + token-budget
    splitting without semantic awareness.
    """
    text = block.text
    tokens = _count_tokens(text)

    # Small block — return as-is
    if tokens <= cfg.max_tokens:
        return [(text, block.section, block.char_start, block.char_end)]

    # Split into sentences
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        # Can't split further — return as-is (oversized but unavoidable)
        return [(text, block.section, block.char_start, block.char_end)]

    # If no embedder, use token-budget-only splitting
    if embedder is None:
        return _split_by_tokens(sentences, text, block, cfg)

    # Semantic splitting with embeddings
    return _split_by_similarity(sentences, text, block, embedder, cfg)


def _split_by_tokens(
    sentences: list[str],
    full_text: str,
    block: Block,
    cfg: ChunkingConfig,
) -> list[tuple[str, Section, int, int]]:
    """Fallback: split sentences on token budget only (no embeddings)."""
    result: list[tuple[str, Section, int, int]] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)
        if current_tokens + sent_tokens > cfg.max_tokens and current:
            chunk_text = " ".join(current)
            c_start, c_end = _find_span(chunk_text, full_text, block.char_start)
            result.append((chunk_text, block.section, c_start, c_end))
            # Overlap
            overlap = current[-cfg.overlap_sentences:] if cfg.overlap_sentences else []
            current = overlap + [sent]
            current_tokens = sum(_count_tokens(s) for s in current)
        else:
            current.append(sent)
            current_tokens += sent_tokens

    if current:
        chunk_text = " ".join(current)
        c_start, c_end = _find_span(chunk_text, full_text, block.char_start)
        result.append((chunk_text, block.section, c_start, c_end))

    return result


def _split_by_similarity(
    sentences: list[str],
    full_text: str,
    block: Block,
    embedder,
    cfg: ChunkingConfig,
) -> list[tuple[str, Section, int, int]]:
    """Split prose using sentence-embedding similarity with rolling mean."""
    # Embed all sentences in one batch
    embeddings = embedder.embed(sentences)  # (N, dim)

    result: list[tuple[str, Section, int, int]] = []
    current: list[str] = [sentences[0]]
    current_tokens = _count_tokens(sentences[0])
    similarities: list[float] = []  # similarities within current chunk

    for i in range(1, len(sentences)):
        sent = sentences[i]
        sent_tokens = _count_tokens(sent)

        # Compute similarity against the sliding window
        window_start = max(0, i - cfg.window_size)
        window_embeddings = embeddings[window_start:i]
        window_mean = np.mean(window_embeddings, axis=0)
        similarity = float(_cosine_sim(embeddings[i], window_mean))

        # Compute rolling mean of similarities in the current chunk
        local_mean = np.mean(similarities) if similarities else 1.0

        # Should we split?
        should_split = (
            (similarity < local_mean * cfg.similarity_drop)
            or (current_tokens + sent_tokens > cfg.max_tokens)
        )

        if should_split and current_tokens >= cfg.min_chunk_tokens:
            # Emit current chunk
            chunk_text = " ".join(current)
            c_start, c_end = _find_span(chunk_text, full_text, block.char_start)
            result.append((chunk_text, block.section, c_start, c_end))

            # Overlap: keep last N sentences
            overlap = current[-cfg.overlap_sentences:] if cfg.overlap_sentences else []
            current = overlap + [sent]
            current_tokens = sum(_count_tokens(s) for s in current)
            similarities = []
        else:
            current.append(sent)
            current_tokens += sent_tokens
            similarities.append(similarity)

    # Flush remaining
    if current:
        chunk_text = " ".join(current)
        c_start, c_end = _find_span(chunk_text, full_text, block.char_start)
        result.append((chunk_text, block.section, c_start, c_end))

    return result


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Atomic block splitting (code / table / list)
# ---------------------------------------------------------------------------

def _split_atomic(
    block: Block,
    result: list[tuple[str, Section, int, int]],
    cfg: ChunkingConfig,
) -> None:
    """Add an atomic block to the result. If too large, split on blank lines."""
    text = block.text
    tokens = _count_tokens(text)

    if tokens <= cfg.max_tokens:
        result.append((text, block.section, block.char_start, block.char_end))
        return

    # For oversized atomic blocks, split on blank lines
    parts = text.split("\n\n")
    current: list[str] = []
    current_tokens = 0

    for part in parts:
        part_tokens = _count_tokens(part)
        if current_tokens + part_tokens > cfg.max_tokens and current:
            chunk_text = "\n\n".join(current)
            c_start, c_end = _find_span(chunk_text, text, block.char_start)
            result.append((chunk_text, block.section, c_start, c_end))
            current = [part]
            current_tokens = part_tokens
        else:
            current.append(part)
            current_tokens += part_tokens

    if current:
        chunk_text = "\n\n".join(current)
        c_start, c_end = _find_span(chunk_text, text, block.char_start)
        result.append((chunk_text, block.section, c_start, c_end))


# ---------------------------------------------------------------------------
# Offset tracking helpers
# ---------------------------------------------------------------------------

def _find_span(chunk_text: str, full_text: str, base_offset: int) -> tuple[int, int]:
    """Find the character offset span of chunk_text within full_text."""
    # Use the first sentence (or first 60 chars) to locate in the original text
    search_key = chunk_text[:60]
    pos = full_text.find(search_key)
    if pos != -1:
        start = base_offset + pos
    else:
        # Fallback: use base offset
        start = base_offset
    end = start + len(chunk_text)
    return start, end
