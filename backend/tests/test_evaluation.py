"""
Guards for the evaluation in evaluation/. The judge decides correctness at run
time, so what can be checked here is the data itself: a ground truth that
disagrees with its own documents, or a document that leaks its version or date
into its text, would make the benchmark's numbers meaningless.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[2] / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

import run_eval  # noqa: E402

QUESTIONS = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))["questions"]
MANIFEST = json.loads((EVAL_DIR / "manifest.json").read_text(encoding="utf-8"))
BY_ID = {q["id"]: q for q in QUESTIONS}


# ── Corpus and ground-truth consistency ──────────────────────────────────────


def doc_text(version: str) -> str:
    entry = next(v for v in MANIFEST["versions"] if v["version"] == version)
    return (EVAL_DIR / "documents" / entry["file"]).read_text(encoding="utf-8")


def valid_versions_on(day: date) -> list[str]:
    """Versions whose window [published, next published) contains `day`."""
    vs = MANIFEST["versions"]
    out = []
    for i, v in enumerate(vs):
        start = date.fromisoformat(v["published_at"])
        end = date.fromisoformat(vs[i + 1]["published_at"]) if i + 1 < len(vs) else None
        if start <= day and (end is None or day < end):
            out.append(v["version"])
    return out


class TestCorpus:
    def test_versions_and_dates_increase(self) -> None:
        vs = MANIFEST["versions"]
        assert len(vs) >= 4
        assert [v["version"] for v in vs] == sorted((v["version"] for v in vs), key=lambda s: [int(p) for p in s.split(".")])
        dates = [v["published_at"] for v in vs]
        assert dates == sorted(dates) and len(set(dates)) == len(dates)

    def test_every_version_is_a_distinct_file(self) -> None:
        # The backend rejects byte-identical uploads (SHA-256 duplicate check).
        texts = [doc_text(v["version"]) for v in MANIFEST["versions"]]
        assert len(set(texts)) == len(texts)

    def test_documents_do_not_reveal_their_own_version_or_date(self) -> None:
        # Version and date are supplied as upload metadata. If the body also said
        # "Version 3.0, effective 2023", the baseline could read it and the
        # comparison would no longer isolate the temporal pipeline.
        for v in MANIFEST["versions"]:
            text = doc_text(v["version"])
            assert "version" not in text.lower(), v["file"]
            assert not re.search(r"\b(19|20)\d{2}\b", text), v["file"]


class TestGroundTruth:
    def test_shape(self) -> None:
        # Deliberately small: every question costs LLM calls on a free API key.
        # Every intent the system implements must still be represented.
        assert 10 <= len(QUESTIONS) <= 15
        assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)
        assert {q["type"] for q in QUESTIONS} == set(run_eval.TYPE_ORDER)
        versions = {v["version"] for v in MANIFEST["versions"]}
        for q in QUESTIONS:
            assert q["question"].strip() and q["answer"].strip(), q["id"]
            assert set(q["supporting_versions"]) <= versions, q["id"]

    def test_version_questions_name_the_version_they_are_answered_from(self) -> None:
        # A version question only tests version pinning if its text actually
        # names a version and the reference answer comes from that one version.
        asked = [q for q in QUESTIONS if q["type"] == "version"]
        assert len(asked) >= 3, "version pinning needs more than a token question"
        for q in asked:
            assert len(q["supporting_versions"]) == 1, q["id"]
            version = q["supporting_versions"][0]
            assert re.search(rf"\bv(?:ersion )?{re.escape(version)}\b", q["question"], re.I), q["id"]

    def test_current_questions_are_answered_by_the_latest_version_only(self) -> None:
        latest = MANIFEST["versions"][-1]["version"]
        for q in QUESTIONS:
            if q["type"] == "current":
                assert q["supporting_versions"] == [latest], q["id"]

    def test_point_in_time_dates_fall_in_the_supporting_version_window(self) -> None:
        for q in QUESTIONS:
            if q["type"] != "point_in_time":
                continue
            found = re.search(r"\d{4}-\d{2}-\d{2}", q["question"])
            assert found, f"{q['id']} has no date"
            assert valid_versions_on(date.fromisoformat(found.group())) == q["supporting_versions"], q["id"]

    def test_numbers_in_reference_answers_appear_in_the_supporting_documents(self) -> None:
        # Catches a reference answer that disagrees with the document it came from.
        for q in QUESTIONS:
            text = " ".join(doc_text(v) for v in q["supporting_versions"]).replace(",", "")
            answer = q["answer"].replace(",", "")
            numbers = set(re.findall(r"\d+", answer))
            numbers = {n for n in numbers if not re.fullmatch(r"(19|20)\d{2}", n)}   # years in the wording
            for n in numbers:
                assert re.search(rf"(?<!\d){n}(?!\d)", text), (q["id"], n)
