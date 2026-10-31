# Temporal RAG with Belief Revision — Architecture Deep Dive

This document explains the **backend** exactly as it is currently implemented —
every claim below is backed by a specific file and function. It assumes you
already know how plain RAG works (embed → retrieve top-k → stuff into a
prompt → generate) and focuses only on what this system adds on top of that.

Read it top to bottom: ingestion first (how documents get their temporal
identity), then the query flow in the actual order the code executes it.

---

## 1. The problem, restated precisely

Normal RAG has no concept of a document changing over time. If you upload:

```
v1.0  "Rahul is the CEO of ABC Ltd."
v2.0  "Rahul resigned in March 2023. Priya became the CEO of ABC Ltd."
v3.0  "Priya continues as the CEO of ABC Ltd."
```

...and ask "who is the CEO?", a plain retriever will happily return all three
chunks ranked by pure text similarity — and v1.0 is often the *cleanest*
textual match for the question, so it can easily outrank the correct answer.

This project attaches **version and time metadata to every chunk**, uses an
LLM to understand *what point in time the question is actually about*, and
uses that understanding to deterministically decide which chunks are even
eligible to answer — before any ranking happens. Separately, it detects when
two retrieved chunks genuinely contradict each other and decides what the LLM
should be told about that.

Three distinct jobs, kept in three distinct stages:

| Job | Question it answers | Stage |
|---|---|---|
| Temporal filtering | Which chunks are allowed to answer *this* question? | hard filter |
| Relevance ranking | Among the allowed chunks, which matches the question best? | soft ranking |
| Conflict arbitration ("belief revision") | Two allowed chunks disagree — what do we tell the LLM? | arbitration |

Note the division of labour between the first two: **time decides
eligibility, relevance decides order.** Nothing about a chunk's age or version
is ever added to its ranking score — §7.2 explains why that would be both
redundant and harmful.

Keeping these separate is the core design idea of the whole system, and it's
worth holding onto as you read the rest of this document.

---

## 2. Backend module map

```
backend/
  routers/
    ingest.py       POST /api/v1/ingest — upload, chunk, embed, resolve lineage
    query.py        POST /api/v1/query  — the full pipeline described in §4–§11
    conflicts.py    GET/POST /api/v1/conflicts — review & manually resolve conflicts
    analytics.py    corpus stats, lineage timeline
    health.py       liveness / index-drift check

  services/
    parser.py              PDF/MD/TXT → plain text + headings
    chunker.py              text → structure-aware chunks (headings, sentence-similarity splits)
    embedder.py              chunk text → 384-d dense vector (MiniLM)
    faiss_store.py           FAISS index (dense vectors; flat inner-product, exact add/search/remove)
    bm25_store.py            BM25 index (keyword)
    version_resolver.py      version lineage: chain guards, supersession, unlinking on delete,
                             valid_from/valid_to, and the one version-comparison policy
    query_analyzer.py        ONE LLM CALL: question → structured temporal intent
    retriever.py              BM25 + FAISS → RRF-fused candidates
    temporal_reranker.py      intent-aware hard filter, then ranking by relevance
    nli.py                    NLI cross-encoder model wrapper
    conflict_detector.py      pair selection → NLI scoring → type classification
    belief_revision.py        conflict arbitration: what reaches the LLM, and how confidently
    context.py                assembles the final prompt string
    llm.py                    Groq client: answer generation (+ shared client for the analyzer)

  models.py    SQLAlchemy models: Document, Chunk, ConflictPair, QueryLog
  config.py    Settings (LLM model, DB, embedding device, data dir)
```

Postgres is the source of truth for everything (documents, chunks, lineage,
resolved conflicts). FAISS and BM25 are derived indexes that can be rebuilt
from Postgres at any time (`scripts/rebuild_indexes.py`).

---

## 3. Ingestion: how a chunk gets its temporal identity

### 3.1 The basic pipeline

`routers/ingest.py :: ingest_document()`:

```
uploaded file + version_string + published_at (+ optional parent_doc_id)
  ↓ version_resolver.validate_lineage(...)   ← read-only guards, §3.2, before
                                               any expensive work is done
  ↓ parser.parse()        → plain text + section headings
  ↓ chunker.chunk()        → ChunkData[] (content, section, char offsets)
  ↓ embedder.embed()        → 384-d vector per chunk
  ↓ INSERT Document, INSERT Chunk rows (not committed yet)
       Chunk.valid_from = published_at
  ↓ version_resolver.apply_lineage(parent_doc_id, ...)  ← the same guards again
                                               under the parent's row lock,
                                               then the supersession, §3.2
  ↓ commit
  ↓ faiss_store.add() + bm25_store.add()            ← index only after commit

deleting a document runs the mirror image:
  ↓ version_resolver.unlink_lineage(...)     ← refuse if a later version
                                               depends on it, else restore the
                                               parent as latest, §3.2
  ↓ DELETE the document (chunks cascade) ↓ commit
  ↓ faiss_store.remove() + bm25_store.remove()      ← exact, after the commit
```

Time is **document-level metadata supplied by the uploader** — `version_string`
and `published_at` are form fields. Nothing is extracted from the document
text itself. If v2.0's PDF says "Rahul resigned in March 2023" and you upload
it with `published_at = 2023-06-01`, the system only knows about June 1st; the
March date inside the sentence is just text.

### 3.2 Version lineage and the chain guards

`services/version_resolver.py :: validate_lineage()` / `apply_lineage()`.

Every document either starts a new lineage (`parent_doc_id = None`) or
extends an existing one by naming its `parent_doc_id`. When it extends one,
three guards must all pass, or the upload is rejected with a `ValueError` (→
HTTP 422 in `ingest.py`):

1. **The parent must be the current latest version of its lineage.**
   You cannot upload v3.0 as a child of v1.0 if v2.0 already exists — that
   would branch the lineage into two "latest" documents.
2. **The new version string must be strictly greater** than the parent's,
   using the shared ordering policy `version_sort_key()` (numeric component
   comparison: `2.10 > 2.9`, `2.2 == 2.2.0`).
3. **The new `published_at` must not be earlier than the parent's.** A
   version dated before its own parent would give the parent a validity
   window that ends before it starts.

These three guards exist for one reason: **query-time filtering trusts that a
lineage is a single, time-ordered chain.** If that invariant ever broke, two
versions could both claim to be valid on the same day, and point-in-time
queries would have no correct answer to fall back to.

The guards run twice: `validate_lineage()` checks them first, before the file
is parsed, chunked and embedded, so a bad submission costs none of that work;
`apply_lineage()` re-runs the same check inside the transaction while holding
the parent's row lock, which is what makes it safe against a concurrent
upload of the same next version. Both call one implementation.

When a new version is accepted, `apply_lineage()` supersedes exactly one
document — the parent:

- `parent.is_latest = False`
- each of the parent's chunks gets
  `Chunk.valid_to = <new document's published_at>` and
  `Chunk.is_superseded = True`
- the **new** document's chunks keep `valid_to = None`,
  `is_superseded = False` — it is now the current version

It does **not** walk the rest of the ancestor chain, because guard 1 has
already made that pointless: the parent is by definition the only version of
the lineage that is still current, so every older version was superseded when
*it* was replaced and already carries `is_latest = False` and a closed window
on every chunk. One lineage, one current version, one document to update.

Deletion (`unlink_lineage()`) is the mirror image and reuses the same
invariant. A document can only be deleted if nothing names it as a parent; for
`v1 → v2 → v3` that means v3 goes and v1 and v2 are refused with a 409. Guard 1
is what makes that rule cheap to state: every superseded version already has
exactly one child, so "has no descendants" and "is the current version" are the
same condition, and a lineage can only be dismantled newest-first. Removing the
current version restores its parent — `is_latest = True` again and
`valid_to = NULL`, `is_superseded = False` on its chunks — which is precisely
the supersession undone. Nothing relinks and no `parent_doc_id` is ever
rewritten, so the chain cannot be left with a hole in it.

### 3.3 The validity window, concretely

For the running example (dates are what `published_at` was set to):

```
v1.0  published 2022-01-10                v2.0  published 2023-03-15
v2.0  published 2023-03-15                v3.0  published 2025-02-01
v3.0  published 2025-02-01                (still current)
```

After all three are ingested, each chunk's window is:

| Version | `valid_from` | `valid_to` | `is_superseded` |
|---|---|---|---|
| v1.0 | 2022-01-10 | 2023-03-15 | True |
| v2.0 | 2023-03-15 | 2025-02-01 | True |
| v3.0 | 2025-02-01 | `None` | False |

The window is **half-open**: `[valid_from, valid_to)`. On the exact day a new
version is published, the new version is the one in force — v2.0's window
does *not* include 2023-03-15; v2.0 starts being valid *on* that day and v1.0
stops being valid *on* that day. (`services/temporal_reranker.py :: valid_on()`
implements this with `start <= day < end`.)

### 3.4 `lineage_roots()` — identifying which lineage a document belongs to

`services/version_resolver.py :: lineage_roots(db, doc_ids)` walks each
document's `parent_doc_id` chain up to its root and returns
`{doc_id: root_doc_id}`. Two documents share a lineage exactly when they map
to the same root. This is computed once per query, over every retrieved
document, and attached to each `CandidateChunk` as `lineage_id`
(`services/retriever.py`). It's what lets the conflict detector distinguish
"these are two versions of the same document" from "these are two unrelated
documents that happen to disagree" — see §9.

---

## 4. The query flow, in actual execution order

This is the complete path through `routers/query.py :: query_endpoint()`,
matching the code's own stage numbering:

```
POST /api/v1/query  { query, max_chunks, retrieve_only, no_temporal }
  │
  ├─ 0.  query_analyzer.analyze(query)                 [SKIPPED if no_temporal]
  │       → QueryAnalysis(intent, as_of / start_date+end_date / version)
  │
  ├─ 1.  retriever.hybrid_retrieve(query, k = max_chunks × 3 if temporal else max_chunks)
  │       → CandidateChunk[] ranked by fused BM25+dense relevance (RRF)
  │
  ├─ 1b. temporal_reranker.rerank(candidates, analysis)  [SKIPPED if no_temporal]
  │       → hard-filters by validity window, then re-ranks survivors
  │       → truncate to max_chunks
  │
  ├─ 1c. conflict_detector.detect(candidates[:20], ...)  [SKIPPED if no_temporal]
  │       → NLI-scored, lineage-typed ConflictResult[]
  │
  ├─ 1d. belief_revision.revise(candidates, conflicts, analysis)
  │       → include/exclude chunk_ids, notices, answer_confidence
  │       (always runs; with 0 conflicts it's a harmless pass-through, which
  │        is exactly what the baseline needs)
  │
  ├─ 2.  context.build_context(chunks, revision, temporal_annotations, time_frame)
  │       → prompt string + which chunk_ids actually fit the token budget
  │
  ├─ 3.  llm.generate(context, query, temporal=...)      → final answer text
  │
  ├─ 4.  write QueryLog row
  └─ 5.  return QueryResponse (answer + per-source scores + analysis + conflicts)
```

The next seven sections walk each numbered stage in detail, in this same
order.

---

## 5. Stage 0 — Temporal Query Analyzer

**File:** `services/query_analyzer.py`. **One LLM call, JSON-mode, validated.**
No regex, no date-parsing library in front of it — the LLM is solely
responsible for understanding what the question means in time; deterministic
code is solely responsible for acting on that understanding.

### 5.1 What it asks for

The model (same Groq model configured for answers, `temperature=0`,
`response_format={"type": "json_object"}`) is given the question plus
`TODAY: <date>` and asked to return exactly:

```json
{"intent": ..., "as_of": ..., "start_date": ..., "end_date": ..., "version": ...}
```

| `intent` | meaning | fields it uses |
|---|---|---|
| `current` | the present state | — |
| `point_in_time` | one specific moment | `as_of` |
| `range` | a period (inclusive, either bound may be open) | `start_date` / `end_date` |
| `version` | names a document version explicitly | `version` |
| `historical` | the past, no specific date/version given | — |

A question whose answer does not depend on time at all ("what does the leave
policy cover?") is a `current` question. It used to have its own `atemporal`
intent, which filtered and ranked identically to `current` and so produced no
different behaviour; it was removed rather than kept as a second name for one
thing.

Relative dates ("last year", "two months ago") are resolved by the LLM
against `TODAY`, which is passed in the prompt.

### 5.2 Validation (`_TemporalParse`, a Pydantic model)

The raw JSON is parsed and validated before anything downstream sees it:

- `intent` must be one of the five known values.
- Dates must be ISO strings (`"2023-01-01"`) — a bare number like `2023` is
  explicitly rejected rather than silently parsed as a Unix timestamp.
- Each intent's *required* field must be present: `point_in_time` needs
  `as_of`; `range` needs at least one of `start_date`/`end_date` and, if both
  are given, `start_date <= end_date`; `version` needs a version string
  containing a digit.
- Fields **irrelevant** to the chosen intent are cleared, not rejected — a
  model that fills `as_of` on a `current` question has still classified the
  question correctly, so that field is just dropped.
- `version` is normalized: `"V1_13"` → `"1.13"`, `"version 2.0"` → `"2.0"`,
  `"release 3.1"` → `"3.1"`.

If parsing or validation fails for any reason (malformed JSON, missing
required field, network error, timeout, missing API key), `analyze()` **never
raises** — it returns the default analysis (`intent=current, source="default",
error="<why>"`), which is exactly how a question with no temporal signal has
always been treated. The UI surfaces this via `QueryAnalysisInfo.source` and
`.error`.

Successful results are cached per `(query_text, today)`, so repeated
questions and evaluation reruns cost one call and parse identically.

### 5.3 What it explicitly does NOT do

It never sees a chunk. It never decides which chunks are valid, relevant,
contradictory, or preferred. It produces one JSON object; every decision
after that is deterministic code. This separation is deliberate: a
misclassified intent changes *which time window is searched*, but it can
never inject a fabricated fact into the answer.

### 5.4 `QueryAnalysis` — the object everything downstream reads

```python
@dataclass(frozen=True)
class QueryAnalysis:
    intent: TemporalIntent = CURRENT
    as_of: date | None = None          # point_in_time
    start_date: date | None = None     # range
    end_date: date | None = None       # range
    version: str | None = None         # version
    source: str = "default"            # "llm" | "default"
    error: str | None = None
```

That is the whole object: five plain fields plus how it was produced. It has
no derived properties — `intent` *is* the decision, and the one place that
acts on it is the filter in §7.1, which reads it directly.

---

## 6. Stage 1 — Hybrid retrieval, with over-fetching

**File:** `services/retriever.py :: hybrid_retrieve(query, db, k)`.

This stage is completely temporally blind — it's the same hybrid retrieval a
non-temporal RAG system would use:

1. **BM25** keyword search → top candidates + raw scores.
2. **Dense** search via FAISS (`IndexFlatIP` — exhaustive inner product; over
   the unit vectors the embedder produces, that IS cosine similarity, so the
   score needs no conversion) → top candidates + similarities.
3. **Reciprocal Rank Fusion**: `score = 1/(60+rank_bm25) + 1/(60+rank_dense)`,
   missing from one retriever = worst-rank penalty for that term.
4. Fetch chunk + document metadata from Postgres for the fused top candidates
   (including `valid_from`, `valid_to`, `is_superseded`, `version_string`,
   `is_latest`, and now `lineage_id` via `version_resolver.lineage_roots()`).
5. `relevance_score` = `rrf_score` min-max normalized to `[0, 1]` across the
   returned set — this is the number the ranking actually uses; raw RRF values
   are tiny (≤ 2/61 ≈ 0.033) and unusable as a UI bar.

### Over-fetching

`routers/query.py` requests `k = max_chunks × 3` when the temporal pipeline
is active (`OVERFETCH_FACTOR = 3`), and exactly `max_chunks` in baseline mode.

**Why:** the temporal filter in the next stage can eliminate *most* of the
retrieved pool — every superseded version for a `current` question, or
everything outside the requested date window for a dated one. If retrieval
only fetched `max_chunks` and the filter then discarded half of them, the LLM
could end up answering from far fewer sources than requested, or from none at
all, even though plenty of valid chunks existed in the corpus. Over-fetching
gives the filter a large enough pool that truncation to `max_chunks` happens
*after* filtering, not before.

---

## 7. Stage 1b — Temporal filtering and ranking

**File:** `services/temporal_reranker.py :: rerank(candidates, query_analysis, now)`.
This is the core "temporal" logic of the whole project, and it is two separate
steps: a hard filter on time, then a ranking that ignores time entirely.

### 7.1 Step 1 — FILTER (hard): is this chunk allowed to answer at all?

```python
def is_temporally_valid(chunk, query_analysis, today) -> bool:
    intent = query_analysis.intent
    if intent is CURRENT:
        return valid_on(chunk, today)   # the window alone; see below
    if intent is POINT_IN_TIME:
        return valid_on(chunk, query_analysis.as_of)
    if intent is RANGE:
        return overlaps(chunk, query_analysis.start_date, query_analysis.end_date)
    if intent is VERSION:
        return version_matches(chunk.version_string, query_analysis.version)
    # HISTORICAL:
    return True
```

| Intent | Chunk is kept when | Notes |
|---|---|---|
| `current` | its window contains today (`valid_on(chunk, today)`) | undated chunks are always valid; `is_superseded` is not consulted (see below) |
| `point_in_time` | its window contains `as_of` | `valid_from <= as_of < valid_to` |
| `range` | its window **overlaps** `[start_date, end_date]` | either bound may be open (`None`) |
| `version` | its version names the same release as the one asked for | `version_matches()`, §7.3 |
| `historical` | always | every version is eligible — "who used to be CEO?" needs all of them |

This is a **hard** filter — a chunk either passes or it's gone. There is no
partial credit; a chunk one day outside the requested range is filtered out
just as completely as one ten years outside it.

Why `current` reads only the window and not `is_superseded`: the flag is set
the moment a newer version is *ingested*, but `published_at` may be in the
future. If v2 is uploaded today dated next month, v1's chunks are flagged at
once, yet v1 is still the policy in force until v2's date — and its
`valid_to` (v2's `published_at`) already says exactly that. Filtering on the
flag would hide v1 while `valid_on` hides the not-yet-valid v2, leaving the
lineage with no current answer at all. The flag stays in the schema for the
lineage bookkeeping in `version_resolver` and for display.

### 7.2 Step 2 — RANK (soft): among the survivors, order them

```
every intent:
    rank by relevance_score, descending, chunk_id breaking ties
```

`relevance_score` is **re-normalized after filtering** (`assign_relevance_scores`
is called again, on the surviving subset) — so the best surviving chunk always
scores 1.0, reflecting competition among the chunks that are actually eligible,
not the larger pre-filter pool. The `chunk_id` tiebreaker makes the order
deterministic and reproducible run-to-run, which matters for evaluation.

**Why there is nothing temporal in the ranking.** An earlier version of this
module scored `current` questions with a weighted composite,
`0.5·relevance + 0.3·freshness + 0.1·version_boost + 0.1·is_latest`. Three of
those four terms could not affect the order at all, and the fourth was
actively harmful:

- `is_latest` — the `current` filter keeps exactly one version per lineage:
  the one whose window contains today. That is the latest document, or (while
  a future-dated newer version is waiting for its publication date) its
  parent, which is precisely the version that should win. Either way the
  filter has already chosen, so the term had nothing left to decide.
- `version_boost` — the analyzer's schema clears `version` for every intent
  except `version` itself, so on a `current` question there is no version to
  prefer and the term was the constant "no preference" value, 0.3.
- `freshness` (`2 ** (-age_days / half_life)`) — within a lineage only one
  version survives a `current` filter, so this was constant too. Across
  lineages it did vary, and that was the problem: it let a *less relevant*
  chunk from an unrelated but newer document outrank a better one, for no
  reason the data supports. The filter has already established which chunks
  are currently in force; being published more recently than an unrelated
  document says nothing about answering the question better.

A weighted sum of one real signal and three constants is that one signal with
extra arithmetic, so the composite, its weights, the half-life setting and the
per-chunk `temporal_score` / `version_boost` / `latest_bonus` /
`composite_score` fields were all removed. Every intent now ranks the same way,
which is also why the code no longer needs a per-intent scoring branch.

For past-facing intents the reasoning is the older and more important one:
applying a freshness bonus to "who was CEO in 2023?" drags the ranking toward
whichever surviving chunk is newest, even though the filter has already
restricted the pool to exactly the chunks valid during 2023 — the recency term
would fight the filter and could still push the wrong chunk to the top.

### 7.3 `version_matches()` — one version policy, in one place

`services/version_resolver.py :: version_matches(chunk_version, requested)` is
the filter for `version` questions, and it is defined in terms of the same
`version_sort_key()` that orders a lineage during ingestion: two versions match
when their sort keys are equal. So `"2"`, `"2.0"` and `"2.0.0"` all name one
release, `"2.1"` does not match `"2.10"` (different numbers, not a string
prefix), and a missing or unparseable version matches nothing.

Keeping it in `version_resolver.py` means the project has exactly one piece of
code that knows how to read a version string. It is used *only* as a filter;
nothing scores a chunk by how well its version matches.

---

## 8. Stage 1c — NLI conflict detection

**File:** `services/conflict_detector.py :: detect(candidates, db, query_id)`.
Runs on the **top 20** post-filter, post-truncation candidates
(`CONFLICT_SCAN_LIMIT`).

```
1. Pair selection    _select_pairs(): candidates from DIFFERENT documents that
                      either share a named entity (CamelCase/version/ALL_CAPS
                      regex) OR have lexical (bag-of-words) cosine > 0.55.
                      Cuts an O(n²) pair set down to a manageable few.

2. Cache check        query conflict_pairs for already-scored canonical pairs
                       (min(id), max(id)) — avoids re-running NLI.

3. Batch NLI          nli.predict([(text_a, text_b), ...]) → P(contradiction)
                       per uncached pair (cross-encoder/nli-deberta-v3-small).

4. Classify           _classify(a, b, score) → type or None (§9)

5. Persist            INSERT ... ON CONFLICT DO NOTHING (idempotent, safe
                       under concurrent requests)
```

Same-document pairs are **deliberately excluded** at the pair-selection step.
Two passages of one document are neither a version change nor a disagreement
between sources, and belief revision has no action it could take on such a
pair — in practice, NLI cross-encoders flag topically adjacent prose within
one long document as "contradictory" fairly often, and because the final
confidence takes the *worst* result across all conflicts, those false
positives would drag a genuinely well-resolved answer down to "low
confidence" for no reason.

---

## 9. Conflict classification: lineage decides the type, not dates

**File:** `services/conflict_detector.py :: _classify()`.

```python
def _classify(a, b, nli_score) -> str | None:
    if nli_score < NLI_THRESHOLD:      # 0.70
        return None
    if _lineage(a) == _lineage(b):
        return "version_supersession"
    return "direct_contradiction"
```

This is a direct rewrite of an earlier, weaker rule. Previously, the type was
decided purely by publication-date gap (`>= 90 days apart → "version
supersession"`), which meant two *completely unrelated* documents published a
few months apart would get mislabeled as versions of each other and then be
resolved with unearned high confidence.

Now the type is decided by **lineage**, using `lineage_id` (from
`version_resolver.lineage_roots()`, attached to every `CandidateChunk` at
retrieval time, §3.4):

- **`version_supersession`** — both chunks share a lineage root: they are
  literally different versions of the same document. The "conflict" is the
  document itself changing over time. Publication dates play no role in this
  decision at all.
- **`direct_contradiction`** — the chunks come from unrelated documents (no
  shared lineage). Nothing in the corpus establishes which one is
  authoritative; this is a genuine disagreement between independent sources.

Cached rows are **reclassified on every load** (`_reclassify_cached()`):
the NLI score is what's cached (expensive to recompute); the type is cheap to
re-derive from current lineage and is corrected in the database if it has
changed, so a row written under the old date-gap rule never leaks a stale
type into a live query.

---

## 10. Stage 1d — Belief revision (conflict arbitration)

**File:** `services/belief_revision.py :: revise(candidates, conflicts, db, query_analysis)`.

Despite the name, this stage does not maintain any formal "belief" state — it
is a **rule-based arbitration step** that decides, for each detected
conflict, what the LLM is told and how confidently the final answer may
speak. It runs on every query (harmlessly, as a pass-through, when there are
zero conflicts) and never touches the database except to look up a
human-recorded resolution.

### 10.1 The one invariant that governs everything

> **Evidence is only ever excluded for questions about the present**
> (`intent is current`). For every other intent, the temporal
> filter (§7.1) has already restricted the candidates to the exact time
> window the question is about — so a disagreement *inside* that window is
> something to **explain to the LLM**, never something to delete.

This is why "who was CEO in 2023?" can legitimately need *both* the chunk
naming Rahul and the chunk naming Priya — the filter already proved both are
correct for part of 2023; belief revision's job here is only to make sure the
LLM is told *when* each one applies, not to pick a winner.

### 10.2 The three rules

**Rule 1 — `version_supersession` (same lineage, i.e. versions of one document):**

Always keep both, always high confidence. The notice tells the LLM the exact
validity window of each version:

```
ℹ VERSION CHANGE: ABC Handbook v1.0 (valid 2022-01-10 → 2023-03-15) and
ABC Handbook v2.0 (valid 2023-03-15 → 2025-02-01) are versions of the same
document and say different things. Both are included because each is
correct for its own validity period. Answer from the version(s) valid for
the time the question asks about, and say when it changed.
```

Neither chunk is marked `← PREFERRED` — a preference label would imply one
side was discarded, and here both are simply correct for different periods.
For a `current` question this rule cannot fire: a lineage's validity windows
follow each other without overlapping, so only one version is valid today.
It matters for `historical` and `range` questions where several versions
survive the filter. For the same reason version changes are never reviewed
(§10.3): no decision about them could change what reaches the LLM.

**Rule 2 — `direct_contradiction` about the present** (unrelated documents,
question intent is `current`):

```python
if present and gap_days > CONTRADICTION_RECENCY_GAP_DAYS:   # 90, fixed heuristic
    exclude the older chunk, prefer the newer  →  medium confidence
else:
    keep both, flag the disagreement           →  low confidence
```

Only here — genuinely unrelated sources, asked about right now — is a chunk
ever excluded automatically. If the newer of the two was published more than
90 days after the older, the older is dropped from the context but **quoted
in a footnote** (capped at 400 characters) so the LLM can still describe what
changed if asked. If the two are close in time (≤ 90 days) or either is
undated, neither side can be confidently preferred, so both stay and the
confidence drops to `low`.

**Rule 3 — `direct_contradiction` about the past** (any intent other than
`current`):

Always keep both, always `low` confidence — recency can't be used to prefer
one source over another when the question isn't about the present in the
first place.

### 10.3 Stored (human) resolutions override the automatic rules

Only a `direct_contradiction` is reviewed. A reviewer records one explicit
decision on the Conflicts page; it replaces Rules 2–3 for that pair, still
under the present-vs-past invariant:

| Decision | Present question | Past question |
|---|---|---|
| `prefer_source` + `preferred_chunk_id` | exclude the other side, quoted as set aside (high) | keep both; the notice says which side the reviewer trusts where they overlap (high) |
| `both_valid` (different scopes / not a real contradiction) | keep both (high) | keep both (high) |

The winner is the chunk the reviewer picked, never inferred from dates, so it
is well defined when publication dates are identical or missing, and a
reviewer can prefer the *older* source. Past questions keep the rejected side
because a preference for B now does not make A wrong about its own period.
"Not sure" is not a decision: the conflict stays pending and Rules 2–3 apply.
A row whose decision is malformed (`ConflictPair.has_valid_resolution()` is
false) is ignored the same way. Decisions can be changed by resolving again.

The retired types are converted once by `migrations.py`: `temporal_preference`
becomes `prefer_source` naming the later-dated chunk when both dates exist and
differ, `scope_clarification` becomes `both_valid`, and `manual` — or a
`temporal_preference` whose "newer" side is undefined — is reopened, keeping
the reviewer's note.

### 10.4 Aggregation

- **`answer_confidence`** is the **worst** confidence level across every
  conflict processed for this query (`high` > `medium` > `low`).
- **`preferred_chunks`** contains only chunks that *won* a conflict — i.e.
  the other side was actually excluded in their favor. A chunk from a
  `version_supersession` pair, or one from an unresolved `low`-confidence
  contradiction, is never marked preferred, because no ruling was actually
  made in its favor.
- **`exclusion_reasons`** gives a plain-English reason per excluded chunk,
  surfaced to the UI per source card.

---

## 11. Stage 2 & 3 — Context construction and final answer

### 11.1 `services/context.py :: build_context(chunks, revision_result, temporal_annotations, time_frame)`

Builds the literal prompt text, token-budgeted (`CONTEXT_TOKEN_BUDGET = 3000`,
counted with `tiktoken`). Two very different modes controlled by
`temporal_annotations`:

**Temporal mode** (`temporal_annotations=True`, the normal pipeline):

```
QUESTION TIME FRAME: 2023-01-01 to 2023-12-31. Answer for this time
frame, not necessarily for the present.

ℹ VERSION CHANGE: ABC Handbook v1.0 (valid 2022-01-10 → 2023-03-15) and ...

[SOURCE 1 | ABC Handbook | v1.0 | valid 2022-01-10 → 2023-03-15]
Rahul is the CEO of ABC Ltd.

[SOURCE 2 | ABC Handbook | v2.0 | valid 2023-03-15 → 2025-02-01]
Rahul resigned in March 2023. Priya became the CEO of ABC Ltd.

ANSWER CONFIDENCE: HIGH
REASON: 1 conflict(s), each explained by the document's version history ...
```

- `time_frame` (from `temporal_reranker.describe_time_frame()`) is emitted
  first, and is `None` for `current` questions — there's nothing
  useful to tell the model beyond "answer for the present," which is the
  default framing anyway.
- Each source header shows its validity window: `valid A → B` (was superseded
  on B) or `valid from A` (still current).
- A chunk that *won* a conflict is tagged `← PREFERRED`; one that was
  excluded never appears here at all (it's been filtered out of
  `context_chunks` in `query.py` before this function is even called) but is
  quoted in the conflict notice instead.

**Baseline mode** (`temporal_annotations=False`):

```
[SOURCE 1 | ABC Handbook]
Rahul is the CEO of ABC Ltd.

[SOURCE 2 | ABC Handbook]
Rahul resigned in March 2023. Priya became the CEO of ABC Ltd.

[SOURCE 3 | ABC Handbook]
Priya continues as the CEO of ABC Ltd.
```

No version, no date, no notice, no confidence footer — see §13 for why this
matters.

### 11.2 `services/llm.py :: generate(context, query, temporal=True)`

Calls the configured Groq model with one of two **different system prompts**:

- `SYSTEM_PROMPT` (temporal mode) explicitly instructs the model to read
  `valid A → B` windows, honor a `QUESTION TIME FRAME` line if present,
  prefer `← PREFERRED` sources, and lower its confidence when told to.
- `BASELINE_SYSTEM_PROMPT` says nothing about time, versions, or validity at
  all — just "answer from the sources, cite them, don't invent facts."

---

## 12. Worked examples: same documents, different evidence

All three examples use the exact same ingested corpus:

```
v1.0  valid 2022-01-10 → 2023-03-15   "Rahul is the CEO of ABC Ltd."
v2.0  valid 2023-03-15 → 2025-02-01   "Rahul resigned in March 2023.
                                        Priya became the CEO of ABC Ltd."
v3.0  valid 2025-02-01 → (current)    "Priya continues as the CEO of ABC Ltd."
```

### "Who is the CEO now?"

- Analyzer → `intent=current`
- Filter → keeps only **v3.0** (`valid_on(today)`)
- Ranking → relevance; only one candidate survives anyway
- Conflicts → none possible (only one chunk survives the filter)
- Context → `[SOURCE 1 | ABC Handbook | v3.0 | valid from 2025-02-01]`
- **Answer: "Priya is the CEO."**

### "Who was the CEO in 2023?"

- Analyzer → `intent=range, start_date=2023-01-01, end_date=2023-12-31`
- Filter → keeps **v1.0** (valid until 2023-03-15, overlaps the range) and
  **v2.0** (valid from 2023-03-15, overlaps the range); **drops v3.0**
  (not valid until 2025)
- Ranking → relevance, with nothing temporal added — so the order cannot be
  pulled toward whichever version happens to be newer
- Conflicts → v1.0 and v2.0 share a lineage → `version_supersession` → both
  kept, high confidence, each labeled with its window
- Context → both sources, plus `QUESTION TIME FRAME: 2023-01-01 to
  2023-12-31...`
- **Answer: "Rahul was CEO until March 2023, when Priya took over."**

### "Who was the CEO according to v2?"

- Analyzer → `intent=version, version="2"`
- Filter → keeps only chunks whose version matches "2" (i.e. **v2.0**),
  via `version_matches()`
- Ranking → relevance
- Conflicts → none (only one chunk survives)
- Context → `QUESTION TIME FRAME: version 2 of the documents...`, then
  `[SOURCE 1 | ABC Handbook | v2.0 | valid 2023-03-15 → 2025-02-01]`
- **Answer: "According to version 2.0, Priya is the CEO"** (v2.0's text says
  Rahul resigned and Priya took over — the answer reflects what v2.0 itself
  states, not necessarily the very latest reality)

### A fourth example: an unrelated contradicting document

Add a standalone press release, **not** part of the handbook's lineage,
published 2025-06-01: *"Arjun is the CEO of ABC Ltd."*

**"Who is the CEO now?"** — filter keeps v3.0 (current handbook) and the
press release (also current, unrelated lineage). NLI flags them as
contradictory; `_classify()` sees they don't share a lineage →
`direct_contradiction`. The press release is ~120 days newer than v3.0
(> 90-day threshold) → v3.0 is excluded, quoted in a footnote, press release
preferred, confidence `medium`.

**"Who was the CEO in 2025?"** — same two chunks survive the filter (both
valid during 2025), but the intent is `range`, not `current`, so the
"exclude the older" rule never applies (§10.1's invariant). Both are kept,
confidence drops to `low`, and the LLM is told to present both claims and
recommend verifying against primary sources.

---

## 13. Baseline mode (`no_temporal=true`)

`routers/query.py` skips stages 0, 1b (filter+rerank), 1c (conflict
detection), and passes `revision_result=None` to context building. What
survives:

- **Retrieval** — the same `hybrid_retrieve()`, but at exactly `max_chunks`
  (no over-fetch, since there's no filter to compensate for).
- **Belief revision is still *called*** with an empty conflict list — this is
  intentional: with zero conflicts it's a pure pass-through
  (`include_chunks = all`, `answer_confidence = "none"`), which is exactly
  the behavior a baseline needs, so there's no need for a separate code path.
- **Context** — `temporal_annotations=False`: headers are stripped to just
  `[SOURCE N | title]`, no validity windows, no time-frame line, no conflict
  notices, no confidence footer.
- **System prompt** — `BASELINE_SYSTEM_PROMPT`, which never mentions time,
  versions, or validity.

This matters because an LLM told "here are three dated, versioned sources, be
temporally aware" can sometimes infer the right answer from the dates alone
— which would make the baseline look artificially smart and hide the actual
contribution of the temporal pipeline. Baseline mode is designed so the *only*
difference between it and the full pipeline is genuinely the filtering,
reranking, and arbitration logic — not extra hints slipped into the prompt.

One caveat worth knowing: if a *document's own text* mentions a date ("Rahul
resigned in March 2023"), that's part of the chunk content and reaches the
baseline too — content is never redacted. What's suppressed is metadata the
system computed (versions, validity windows, notices), not the source text
itself.

---

## 14. What happens when temporal analysis is ambiguous or fails

There is no separate "ambiguous" state — every failure mode collapses to the
same default: **treat the question as `intent=current`**, exactly as if the
LLM had said so explicitly. This happens when:

- the Groq API call errors (timeout, network, no API key configured)
- the model returns text that isn't valid JSON
- the JSON is valid but fails schema validation (unknown intent, a
  `point_in_time` with no `as_of`, a `range` with `start > end`, a `version`
  string with no digits, a bare-number date, etc.)

In every case, `QueryAnalysis(intent=CURRENT, source="default", error="<reason>")`
is returned, `analyze()` never raises, and the query pipeline proceeds exactly
as it would for any ordinary present-tense question. The API response's
`analysis.source` field is `"default"` (vs. `"llm"` on success, or
`"skipped"` in baseline mode) and `analysis.error` carries the reason, so the
frontend can show *"the question could not be interpreted, so it was treated
as a question about the current state"* rather than silently guessing.

---

## 15. Summary: why three stages, not one

It's worth being explicit about why filtering, reranking, and arbitration
aren't collapsed into a single scoring formula:

- **Filtering** (§7.1) is a **correctness** constraint — a chunk from 2022
  literally cannot answer "what's true in 2025?" as a matter of definition,
  not a matter of relevance. This must be a hard boolean, or a sufficiently
  irrelevant-but-recent chunk could still leak through.
- **Ranking** (§7.2) is a **preference** among chunks that are all equally
  *valid* — once the filter has run, the only thing left to separate them is
  how well each matches the question, so relevance alone decides the order.
- **Arbitration** (§10) operates on a different axis entirely: it doesn't ask
  "is this chunk valid" or "is this chunk relevant" — both are already
  settled by the time it runs — it asks "these two *valid, relevant* chunks
  disagree; what does the LLM need to know about that disagreement?" This is
  content-driven (NLI), not metadata-driven, and it's the only stage that
  can ever *exclude* evidence outright — and even then, only for
  present-tense questions, and only between unrelated sources with a clear
  publication gap.

Collapsing these into one formula would mean a single number has to encode
"is this even a legal answer," "how good an answer is it," and "should this
be trusted over a contradicting alternative" all at once — which is exactly
the kind of single-score design that made the original recency-only reranker
unable to correctly answer a question like "who was CEO in 2023?" in the
first place.

The corollary, and the reason §7.2 ranks on relevance alone: once the stages
*are* separated, the ranking stage has no business re-stating what the filter
already decided. Any "prefer the newer / latest / matching version" term at
that point is either constant across the survivors — the filter having removed
everything else — or it is quietly overruling the filter.
