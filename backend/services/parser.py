"""
services/parser.py — Document parsing service.

parse(file_bytes, filename) → ParseResult

Supports:
  - PDF  (.pdf)  — page-by-page text extraction via PyMuPDF; bookmark tree → headings
  - MD   (.md)   — decode UTF-8; extract # headings
  - TXT  (.txt)  — decode UTF-8; no headings
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF


@dataclass
class Heading:
    """Represents a document heading with its position in the full text."""
    level: int        # 1–6 (from # count or PDF bookmark depth)
    title: str
    char_offset: int  # character position in the full extracted text


@dataclass
class ParseResult:
    text: str
    headings: list[Heading] = field(default_factory=list)
    source_type: str = "txt"   # "pdf" | "md" | "txt"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def parse(file_bytes: bytes, filename: str) -> ParseResult:
    """
    Parse raw file bytes into plain text + a list of headings.

    Raises:
        ValueError: if the file extension is not supported.
    """
    ext = _extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        return _parse_pdf(file_bytes)
    elif ext == ".md":
        return _parse_markdown(file_bytes)
    else:
        return _parse_text(file_bytes)


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------

def _parse_pdf(file_bytes: bytes) -> ParseResult:
    """Extract text page-by-page and pull headings from the bookmark tree."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text("text"))  # type: ignore[arg-type]

    full_text = "\n\n".join(pages)

    # Build headings from TOC (table of contents / bookmarks)
    headings: list[Heading] = []
    toc = doc.get_toc()  # [[level, title, page], ...]
    if toc:
        # Map TOC entries to character offsets by searching the text
        for level, title, _page in toc:
            stripped = title.strip()
            if not stripped:
                continue
            pos = full_text.find(stripped)
            offset = pos if pos != -1 else 0
            headings.append(Heading(level=level, title=stripped, char_offset=offset))

    doc.close()
    return ParseResult(text=full_text, headings=headings, source_type="pdf")


def _parse_markdown(file_bytes: bytes) -> ParseResult:
    """Decode markdown, extract ATX headings (#, ##, …)."""
    text = file_bytes.decode("utf-8", errors="replace")

    headings: list[Heading] = []
    for match in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE):
        level = len(match.group(1))
        title = match.group(2).strip()
        headings.append(Heading(level=level, title=title, char_offset=match.start()))

    return ParseResult(text=text, headings=headings, source_type="md")


def _parse_text(file_bytes: bytes) -> ParseResult:
    """Plain text — no heading extraction."""
    text = file_bytes.decode("utf-8", errors="replace")
    return ParseResult(text=text, headings=[], source_type="txt")


def _extension(filename: str) -> str:
    """Return the lowercase file extension including the dot, e.g. '.pdf'."""
    idx = filename.rfind(".")
    if idx == -1:
        return ""
    return filename[idx:].lower()
