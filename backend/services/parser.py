"""
services/parser.py — Document parsing service.

parse(file_bytes, filename) → ParseResult

Supports:
  - PDF  (.pdf)  — page-by-page text extraction via PyMuPDF; bookmark tree → headings
  - MD   (.md)   — decode UTF-8; extract # headings
  - TXT  (.txt)  — decode UTF-8; implicit heading detection via structural patterns

Heading priority:
  1. Explicit headings from Markdown # markers or PDF TOC bookmarks.
  2. Only if unavailable, fall back to generic regex-based implicit detection.
  Implicit detection NEVER runs on Markdown files.
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
        result = _parse_pdf(file_bytes)
        # Fallback: if PDF had no bookmarks/TOC, detect implicit headings
        if not result.headings:
            result.headings = _detect_implicit_headings(result.text)
        return result
    elif ext == ".md":
        # NEVER run implicit detection on Markdown.
        # # markers are the authoritative structure.
        # If the user wrote no # markers, the document genuinely has
        # no sections — don't invent them.
        return _parse_markdown(file_bytes)
    else:
        result = _parse_text(file_bytes)
        result.headings = _detect_implicit_headings(result.text)
        return result


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


# ---------------------------------------------------------------------------
# Implicit heading detection (for TXT and empty-TOC PDFs)
# ---------------------------------------------------------------------------

# Patterns that look like structural section boundaries:
_IMPLICIT_HEADER_PATTERNS = [
    re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$"),         # Title Case multi-word: "Introduction To ML"
    re.compile(r"^[A-Z][a-z]+(?:\s+[A-Za-z]+)*:\s*$"),       # Trailing colon: "Model architecture:"
    re.compile(r"^[A-Z][a-z]+(?:\s+[A-Za-z]+)*\?\s*$"),      # Question heading: "What is AI?"
    re.compile(r"^\d+(?:\.\d+)*\s+[A-Z]"),                   # Numbered section: "1.2 Background"
]


def _is_implicit_header(line: str) -> bool:
    """Return True if a plain-text line looks like a structural heading."""
    line = line.strip()

    if not line or len(line) > 80:
        return False

    # Lines ending with sentence-terminal punctuation are not headings
    if line.endswith((".", ",", ";", "!")):
        return False

    # ALL CAPS: at least 4 chars, alphabetic (catches ABSTRACT, INTRODUCTION, etc.)
    if line.isupper() and len(line) >= 4 and line.replace(" ", "").isalpha():
        return True

    for pattern in _IMPLICIT_HEADER_PATTERNS:
        if pattern.match(line):
            return True

    return False


def _detect_implicit_headings(text: str) -> list[Heading]:
    """
    Scan plain text for structural markers that behave like headings.

    Used only as a fallback when the primary parser returns no explicit
    headings (empty PDF TOC or .txt files). Never called on Markdown.
    """
    headings: list[Heading] = []
    offset = 0

    for line in text.split("\n"):
        stripped = line.strip()
        if _is_implicit_header(stripped):
            headings.append(Heading(
                level=1,  # implicit headings are all treated as level 1
                title=stripped,
                char_offset=offset,
            ))
        offset += len(line) + 1  # +1 for the \n

    return headings
