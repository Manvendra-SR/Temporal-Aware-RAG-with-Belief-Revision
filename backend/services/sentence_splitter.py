"""
services/sentence_splitter.py — Sentence boundary detection.

Uses pySBD for abbreviation-safe, decimal-safe, URL-safe sentence splitting.
Wrapped in a thin module so the dependency is isolated — if pySBD is ever
swapped for another library, only this file changes.

Usage:
    from services.sentence_splitter import split_sentences
    sentences = split_sentences("Dr. Smith published v2.3.1. It works.")
    # → ["Dr. Smith published v2.3.1.", "It works."]
"""

from __future__ import annotations

import pysbd

_segmenter = pysbd.Segmenter(language="en", clean=False)


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.

    Handles abbreviations (Dr., e.g., i.e.), decimal numbers (v2.3.1),
    URLs, and other common edge cases that break naïve regex splitters.

    Returns an empty list for blank input.
    """
    if not text.strip():
        return []
    return [s.strip() for s in _segmenter.segment(text) if s.strip()]
