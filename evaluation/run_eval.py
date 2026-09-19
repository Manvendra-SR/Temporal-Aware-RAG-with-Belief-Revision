#!/usr/bin/env python
"""
Run the versioned-policy question set through the RAG pipeline twice — with
temporal reasoning (no_temporal=false) and without it (no_temporal=true) — and
compare both against the ground truth in evaluation/questions.json.

    python evaluation/run_eval.py [--url http://localhost:8000] [--show-failures]

Both runs use the same retrieval stack, the same answer model and the same
max_chunks; the only difference is the temporal pipeline (see "Baseline mode"
in the README).

Grading: an LLM judge compares each answer with the reference answer in
questions.json and returns correct / incorrect.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parent / "backend"

TYPE_ORDER = ["current", "point_in_time", "historical", "conflict", "stable"]


# ─────────────────────────────────────────────────────────────────────────────
# LLM judge
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You grade answers to questions about a company policy that changed over time.

You are given the question, a reference answer, and a candidate answer.

- CORRECT: the candidate agrees with the reference answer on every key fact
  (the values asked for and, for questions about change or history, all the
  values and their order or direction).
- INCORRECT: the candidate gives a different value as its answer; lists several
  conflicting values without committing to one; leaves out a key fact from the
  reference; or says the information is not available.
- Mentioning other values (older or newer) as background is fine, as long as the
  candidate clearly commits to the reference answer for what was asked.
- Ignore citation markers such as [SOURCE 1], formatting, wording, and number
  formatting (60,000 vs 60000, INR vs Rs).

Return ONLY a JSON object: {"verdict": "correct" or "incorrect", "reason": "<one short sentence>"}
"""


def judge_answer(client, model: str, question: str, reference: str, answer: str) -> tuple[bool, str]:
    """Ask the judge model for a verdict. Raises on API or format errors."""
    user = f"QUESTION: {question}\nREFERENCE ANSWER: {reference}\nCANDIDATE ANSWER: {answer}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=1024,
        timeout=60,
    )
    data = json.loads(resp.choices[0].message.content or "")
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in ("correct", "incorrect"):
        raise ValueError(f"unexpected judge verdict: {verdict!r}")
    return verdict == "correct", str(data.get("reason", "")).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Running the pipeline
# ─────────────────────────────────────────────────────────────────────────────


def with_retries(fn, attempts: int = 3, base_delay: float = 4.0):
    """Call fn(); on failure wait 4s, 12s, ... and retry (rate limits, transient 5xx)."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * (3 ** attempt))


def ask(http: httpx.Client, api: str, question: str, *, temporal: bool, max_chunks: int) -> dict:
    def call():
        r = http.post(f"{api}/query", json={"query": question, "max_chunks": max_chunks,
                                             "no_temporal": not temporal})
        r.raise_for_status()
        return r.json()
    return with_retries(call)


def check_corpus(http: httpx.Client, api: str, manifest: dict, allow_extra: bool) -> None:
    """Abort unless the backend holds exactly the five policy versions."""
    items = http.get(f"{api}/documents", params={"limit": 100}).json()["items"]
    expected = {(v["version"], v["published_at"]) for v in manifest["versions"]}
    ours = [d for d in items if d["title"] == manifest["title"]]
    found = {(d["version_string"], (d["published_at"] or "")[:10]) for d in ours}
    extra = len(items) - len(ours)

    problems = []
    if found != expected:
        problems.append("the policy versions are not all ingested (run evaluation/ingest_corpus.py)")
    if extra and not allow_extra:
        problems.append(
            f"the backend also holds {extra} other document(s), which would leak into retrieval "
            "(use the isolated database from evaluation/README.md, or pass --allow-extra-docs)"
        )
    if problems:
        sys.exit("Corpus check failed: " + "; and ".join(problems) + ".")


def used_versions(response: dict) -> list[str]:
    return sorted({s["version_string"] for s in response["sources"]
                   if s.get("used_in_answer") and s.get("version_string")})


def run_question(http, api, q, *, temporal, max_chunks, judge) -> dict:
    """Ask one question one way and grade the answer."""
    rec: dict = {"answer": None, "correct": False, "judge_reason": "", "error": None}
    try:
        resp = ask(http, api, q["question"], temporal=temporal, max_chunks=max_chunks)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"query failed: {exc}"
        return rec

    answer = resp.get("answer")
    if answer and answer.startswith("[LLM unavailable"):
        sys.exit(f"The backend could not call the LLM: {answer}. Set GROQ_API_KEY and restart it.")

    rec.update(
        answer=answer,
        latency_ms=resp.get("latency_ms"),
        used_versions=used_versions(resp),
        sources=[{"version": s.get("version_string"), "used": bool(s.get("used_in_answer"))}
                 for s in resp["sources"]],
        intent=resp["analysis"]["intent"] if temporal else None,
        analysis_source=resp["analysis"]["source"] if temporal else None,
        analysis_error=resp["analysis"].get("error") if temporal else None,
        filter=resp.get("temporal_filter"),
        confidence=resp.get("answer_confidence"),
    )
    if not answer or not answer.strip():
        rec["judge_reason"] = "no answer was returned"
        return rec
    try:
        rec["correct"], rec["judge_reason"] = with_retries(lambda: judge(q["question"], q["answer"], answer))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"judge failed: {exc}"
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def pct(k: int, n: int) -> float:
    return 100.0 * k / n if n else 0.0


def print_report(results: list[dict], n_fallback: int, n_errors: int) -> dict:
    n = len(results)
    normal = sum(r["normal"]["correct"] for r in results)
    temporal = sum(r["temporal"]["correct"] for r in results)
    diff = pct(temporal, n) - pct(normal, n)

    print("\n" + "=" * 52)
    print(f"Normal RAG:   {normal}/{n} ({pct(normal, n):.1f}%)")
    print(f"Temporal RAG: {temporal}/{n} ({pct(temporal, n):.1f}%)")
    print(f"Improvement:  {diff:+.1f} percentage points")
    print("=" * 52)

    fixed = [r["id"] for r in results if r["temporal"]["correct"] and not r["normal"]["correct"]]
    lost = [r["id"] for r in results if r["normal"]["correct"] and not r["temporal"]["correct"]]
    print(f"\nTemporal answered correctly where normal did not: {len(fixed)}  {fixed}")
    print(f"Normal answered correctly where temporal did not: {len(lost)}  {lost}")

    print(f"\n{'type':<15}{'n':>3}   {'Normal':>8}   {'Temporal':>8}")
    for t in TYPE_ORDER:
        rows = [r for r in results if r["type"] == t]
        if rows:
            b = sum(r["normal"]["correct"] for r in rows)
            m = sum(r["temporal"]["correct"] for r in rows)
            print(f"{t:<15}{len(rows):>3}   {b:>5}/{len(rows):<2}   {m:>5}/{len(rows):<2}")

    if n_fallback:
        print(f"\nWARNING: the temporal analyzer fell back to 'current' on {n_fallback}/{n} questions "
              "(LLM call failed or returned invalid JSON); those ran without temporal understanding.")
    if n_errors:
        print(f"WARNING: {n_errors} query or judge call(s) errored and were scored as wrong. "
              "Results are not reliable; rerun (try --delay 3).")
    return {"normal_correct": normal, "temporal_correct": temporal, "n": n,
            "normal_pct": round(pct(normal, n), 1), "temporal_pct": round(pct(temporal, n), 1),
            "improvement_pp": round(diff, 1)}


def print_failures(results: list[dict]) -> None:
    print("\n--- Wrong answers ---")
    for r in results:
        for arm, label in (("normal", "Normal"), ("temporal", "Temporal")):
            a = r[arm]
            if not a["correct"]:
                text = (a["answer"] or "(no answer)").replace("\n", " ")
                print(f"\n{r['id']} [{r['type']}] {r['question']}\n  reference: {r['answer']}\n"
                      f"  {label}: {text[:300]}\n  judge: {a['judge_reason']}")


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8000", help="backend base URL")
    ap.add_argument("--questions", default=str(HERE / "questions.json"))
    ap.add_argument("--max-chunks", type=int, default=20, help="max_chunks sent for BOTH runs")
    ap.add_argument("--judge-model", default=None, help="Groq model for the judge (default: the answer model)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between queries")
    ap.add_argument("--show-failures", action="store_true", help="print every wrong answer")
    ap.add_argument("--allow-extra-docs", action="store_true", help="skip the isolated-corpus check")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))["questions"]
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    api = args.url.rstrip("/") + "/api/v1"

    sys.path.insert(0, str(BACKEND_DIR))
    from config import settings  # noqa: PLC0415 — needs backend/ on sys.path
    judge_model = args.judge_model or settings.llm_model

    try:
        from services.llm import get_client  # noqa: PLC0415
        client = get_client()
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Cannot set up the LLM judge ({exc}). Set GROQ_API_KEY in .env.")
    judge = lambda q, ref, ans: judge_answer(client, judge_model, q, ref, ans)  # noqa: E731

    print(f"Backend: {args.url}   answer model: {settings.llm_model}   "
          f"judge: {judge_model}   max_chunks: {args.max_chunks}")

    results: list[dict] = []
    with httpx.Client(timeout=180) as http:
        try:
            check_corpus(http, api, manifest, args.allow_extra_docs)
        except httpx.ConnectError:
            sys.exit(f"Cannot reach the backend at {args.url}. Start it first (see evaluation/README.md).")

        print(f"Running {len(questions)} questions x 2 ...\n")
        for i, q in enumerate(questions):
            # Alternate which arm goes first so neither is systematically warmer.
            order = (False, True) if i % 2 == 0 else (True, False)
            arms = {}
            for temporal in order:
                arms["temporal" if temporal else "normal"] = run_question(
                    http, api, q, temporal=temporal, max_chunks=args.max_chunks,
                    judge=judge)
                if args.delay:
                    time.sleep(args.delay)
            mark = lambda a: "ok   " if a["correct"] else "WRONG"  # noqa: E731
            intent = arms["temporal"].get("intent") or "?"
            print(f"{q['id']} {q['type']:<14} normal: {mark(arms['normal'])}  "
                  f"temporal: {mark(arms['temporal'])}  (read as: {intent})")
            results.append({"id": q["id"], "type": q["type"], "question": q["question"],
                            "answer": q["answer"], "supporting_versions": q.get("supporting_versions"),
                            **arms})

    n_fallback = sum(r["temporal"].get("analysis_source") == "default" for r in results)
    n_errors = sum(bool(a["error"]) for r in results for a in (r["normal"], r["temporal"]))

    summary = print_report(results, n_fallback, n_errors)
    if args.show_failures:
        print_failures(results)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{stamp}.json"
    out_path.write_text(json.dumps({
        "meta": {"timestamp": datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
                 "backend_url": args.url, "max_chunks": args.max_chunks,
                 "answer_model": settings.llm_model, "judge_model": judge_model,
                 "temporal_analyzer_fallbacks": n_fallback},
        "summary": summary,
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results (every answer, grade and retrieved version): {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
