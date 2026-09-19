import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import axios from 'axios'
import { useShell } from '../components/Layout'

// ── Types ──────────────────────────────────────────────────────────────────

interface SourceResult {
  chunk_id: string
  doc_title: string
  snippet: string
  section_heading: string | null
  rank: number
  // Retrieval scores. See services/retriever.py for the full contract:
  //   bm25_score      raw Okapi BM25 — UNBOUNDED, never render as a fraction
  //   semantic_score  cosine similarity, [0, 1]
  //   rrf_score       rank-fusion score, <= 2/61
  //   relevance_score rrf_score normalised across this query's results, [0, 1]
  bm25_score: number
  semantic_score: number
  rrf_score: number
  relevance_score: number
  version_string: string | null
  published_at: string | null
  /** End of the validity window (when a newer version superseded it); null while current. */
  valid_to: string | null
  is_latest: boolean | null
  is_superseded: boolean
  temporal_score: number | null
  composite_score: number | null
  has_conflict: boolean
  used_in_answer: boolean
  excluded_reason: string | null
}

interface ConflictInfo {
  chunk_id_a: string
  chunk_id_b: string
  conflict_type: string
  nli_score: number
}

type TemporalIntent =
  'current' | 'point_in_time' | 'range' | 'version' | 'historical' | 'atemporal'

interface QueryAnalysisInfo {
  intent: TemporalIntent
  as_of: string | null
  start_date: string | null
  end_date: string | null
  version_hint: string | null
  temporal_qualifier: boolean
  wants_historical_sources: boolean
  source: 'llm' | 'default' | 'skipped'
  error: string | null
}

/** Plain-English description of how the question was interpreted, or null for
 *  a "current" question, which is the unremarkable default. */
function describeAnalysis(a: QueryAnalysisInfo): string | null {
  switch (a.intent) {
    case 'point_in_time':
      return `a question about ${a.as_of}`
    case 'range':
      if (a.start_date && a.end_date) return `a question about ${a.start_date} to ${a.end_date}`
      if (a.end_date) return `a question about the period up to ${a.end_date}`
      return `a question about the period since ${a.start_date}`
    case 'version':
      return `a question about version ${a.version_hint}`
    case 'historical':
      return 'a question about the past'
    case 'atemporal':
      return 'a question that does not depend on time'
    default:
      return null
  }
}

interface TemporalFilterInfo {
  rule: string
  scoring: string
  candidates_retrieved: number
  candidates_valid: number
  candidates_kept: number
}

interface QueryResponse {
  query_id: string
  answer: string | null
  latency_ms: number
  sources: SourceResult[]
  version_hint: string | null
  analysis: QueryAnalysisInfo
  temporal_pipeline_applied: boolean
  temporal_filter: TemporalFilterInfo | null
  sources_used_in_answer: number
  conflicts_detected: number
  conflict_pairs: ConflictInfo[]
  answer_confidence: string | null
  confidence_reason: string | null
  belief_revision_applied: boolean
}

type HistoryEntry = { query: string; latency_ms: number; conflicts: number; ts: string }

// ── Helpers ────────────────────────────────────────────────────────────────

function pct(value: number): string {
  return `${Math.round(value * 100)}%`
}

/**
 * A one-sentence, plain-language explanation of why a source landed where it
 * did. The point is that a reader who has never seen the code can follow the
 * ranking without decoding the numeric bars.
 *
 * Age and version status are deliberately joined with "but" rather than being
 * listed side by side: a document can be both the current version AND old (no
 * newer release exists yet), and reading "is old — from the latest version" as
 * a flat list makes that sound self-contradictory.
 */
function explainRanking(source: SourceResult, temporalApplied: boolean): string {
  const relevance =
    source.relevance_score >= 0.75 ? 'Closely matches the question' :
    source.relevance_score >= 0.35 ? 'Partially matches the question' :
    'Weakly matches the question'

  const version =
    source.is_superseded ? 'from a superseded version' :
    source.is_latest ? 'from the current version' :
    null

  if (!temporalApplied || source.temporal_score === null) {
    return version ? `${relevance}, ${version}.` : `${relevance}.`
  }

  // Each age phrase carries its own verb so it composes cleanly in every case.
  const age =
    source.temporal_score >= 0.75 ? 'was published recently' :
    source.temporal_score >= 0.35 ? 'is moderately dated' :
    'has not been updated in a while'

  if (!version) return `${relevance}; it ${age}.`

  // For the current version, an old date means "nothing newer exists yet"
  // rather than "this is stale", so it is introduced with "though".
  const isOld = source.temporal_score < 0.35
  const joiner = isOld && source.is_latest ? ', though it' : ', and it'
  return `${relevance}, ${version}${joiner} ${age}.`
}

// ── Confidence Badge ───────────────────────────────────────────────────────

function ConfidenceBadge({ level, reason }: { level: string; reason?: string | null }) {
  const cfg =
    level === 'high'   ? { cls: 'badge-success', icon: '●', label: 'High confidence' } :
    level === 'medium' ? { cls: 'badge-warning', icon: '◐', label: 'Medium confidence' } :
    level === 'low'    ? { cls: 'badge-danger',  icon: '○', label: 'Low — verify sources' } :
    null
  if (!cfg) return null
  return (
    <span className={`badge ${cfg.cls}`} title={reason ?? ''} style={{ cursor: reason ? 'help' : 'default' }}>
      {cfg.icon} {cfg.label}
    </span>
  )
}

// ── Score Bar ──────────────────────────────────────────────────────────────
//
// Only ever used for values that genuinely live in [0, 1]. Raw BM25 and RRF
// are shown as plain numbers in the details panel instead — rendering an
// unbounded or tiny-ranged score as a percentage bar was actively misleading
// (BM25 bars sat pinned near full, and RRF was multiplied by an arbitrary 60
// to make it visible at all).

function ScoreBar({ label, value, color, hint }: {
  label: string; value: number; color: string; hint: string
}) {
  return (
    <div className="score-bar-wrap" title={hint}>
      <div className="score-bar-row">
        <span className="score-bar-label">{label}</span>
        <span className="score-bar-val" style={{ color }}>{pct(value)}</span>
      </div>
      <div className="score-bar-track">
        <div
          className="score-bar-fill"
          style={{ width: `${Math.min(100, Math.max(0, value * 100))}%`, background: color }}
        />
      </div>
    </div>
  )
}

// ── Source Card ────────────────────────────────────────────────────────────

function SourceCard({ source, temporalApplied }: { source: SourceResult; temporalApplied: boolean }) {
  const [showDetail, setShowDetail] = useState(false)
  const [expanded, setExpanded] = useState(false)

  const excluded = !!source.excluded_reason
  const unused = !source.used_in_answer && !excluded

  const borderColor = excluded ? 'var(--danger-border)'
    : source.has_conflict ? 'var(--warning-border)'
    : undefined

  return (
    <div className="card" style={{ borderColor, opacity: source.used_in_answer ? 1 : 0.72 }}>
      <div style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: '9px' }}>

        {/* Header: rank, title, version */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px' }}>
          <div style={{ fontWeight: 600, fontSize: '12px', color: 'var(--text-primary)', lineHeight: 1.4, flex: 1 }}>
            <span style={{ color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
              {source.rank}.
            </span>{' '}
            {source.doc_title}
          </div>
          {source.version_string && (
            <span
              className="badge"
              style={{
                background: source.is_latest ? 'var(--success-subtle)' : 'var(--bg-elevated)',
                color: source.is_latest ? 'var(--success)' : 'var(--text-muted)',
                border: `1px solid ${source.is_latest ? 'var(--success-border)' : 'var(--border-med)'}`,
                fontFamily: 'ui-monospace, monospace',
                fontSize: '10px',
                flexShrink: 0,
              }}
            >
              v{source.version_string}
            </span>
          )}
        </div>

        {/* Status line: current vs historical, and whether it informed the answer */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap', fontSize: '10.5px' }}>
          {source.published_at && (
            <span style={{ color: 'var(--text-muted)' }}>
              {source.valid_to
                ? `valid ${source.published_at} → ${source.valid_to}`
                : `valid from ${source.published_at}`}
            </span>
          )}
          {source.is_superseded ? (
            <span style={{ color: 'var(--warning)', fontWeight: 600 }}>Superseded version</span>
          ) : source.is_latest ? (
            <span style={{ color: 'var(--success)', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '3px' }}>
              <span className="dot dot-success" />Current version
            </span>
          ) : null}

          {source.used_in_answer && (
            <span style={{ color: 'var(--accent-hover)', fontWeight: 600 }}>Used in answer</span>
          )}
          {unused && (
            <span
              style={{ color: 'var(--text-muted)' }}
              title="Retrieved and ranked, but it did not fit within the answer's context budget."
            >
              Not used — beyond context limit
            </span>
          )}
        </div>

        {/* Why belief revision dropped it */}
        {excluded && (
          <div className="alert alert-danger" style={{ padding: '6px 10px', fontSize: '11px', gap: '6px' }}>
            <strong>Excluded from the answer.</strong> {source.excluded_reason}
          </div>
        )}

        {/* Snippet */}
        <div
          onClick={() => setExpanded(e => !e)}
          style={{
            fontSize: '11.5px', color: 'var(--text-secondary)', lineHeight: 1.6,
            cursor: 'pointer', display: '-webkit-box',
            WebkitLineClamp: expanded ? 'unset' : 2,
            WebkitBoxOrient: 'vertical', overflow: 'hidden',
          }}
          title="Click to expand"
        >
          {source.snippet}
        </div>

        {/* Plain-language ranking explanation */}
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          {explainRanking(source, temporalApplied)}
        </div>

        {/* The two scores that are genuinely 0-1 and drive the ranking */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
          <ScoreBar
            label="Relevance"
            value={source.relevance_score}
            color="var(--info)"
            hint="How well this chunk matched the question, combining keyword and meaning search. Scaled relative to the other results for this query."
          />
          {temporalApplied && source.temporal_score !== null && (
            <ScoreBar
              label="Recency"
              value={source.temporal_score}
              color="#f97316"
              hint="Freshness weight from the document's date. Halves for every half-life period of age."
            />
          )}
          {temporalApplied && source.composite_score !== null && (
            <ScoreBar
              label="Final score"
              value={source.composite_score}
              color="#a855f7"
              hint="The ranking score: relevance, recency, version match and latest-version bonus combined."
            />
          )}
        </div>

        {/* Conflict flag */}
        {source.has_conflict && (
          <div className="alert alert-warning" style={{ padding: '5px 10px', fontSize: '11px', gap: '6px' }}>
            Contradicts another retrieved source
          </div>
        )}

        {/* Raw diagnostics — deliberately numbers, not bars */}
        <button
          onClick={() => setShowDetail(d => !d)}
          className="btn btn-ghost"
          style={{ padding: '3px 8px', fontSize: '10.5px', alignSelf: 'flex-start' }}
        >
          {showDetail ? 'Hide' : 'Show'} raw scores
        </button>
        {showDetail && (
          <dl className="raw-scores">
            <div><dt>BM25 (keyword)</dt><dd>{source.bm25_score.toFixed(3)}</dd></div>
            <div><dt>Cosine (meaning)</dt><dd>{source.semantic_score.toFixed(3)}</dd></div>
            <div><dt>RRF (fused rank)</dt><dd>{source.rrf_score.toFixed(5)}</dd></div>
            {source.section_heading && (
              <div><dt>Section</dt><dd>{source.section_heading}</dd></div>
            )}
            <p className="raw-scores-note">
              BM25 is unbounded and RRF is capped near 0.033 — neither is a
              percentage, so they are shown as raw values.
            </p>
          </dl>
        )}
      </div>
    </div>
  )
}

// ── Conflict Details Panel ─────────────────────────────────────────────────

function ConflictDetailsPanel({
  sources, conflictPairs, confidenceReason,
}: {
  sources: SourceResult[]
  conflictPairs: ConflictInfo[]
  confidenceReason?: string | null
}) {
  const [open, setOpen] = useState(false)
  if (conflictPairs.length === 0) return null

  const sourceMap = new Map(sources.map(s => [s.chunk_id, s]))

  return (
    <div style={{ borderTop: '1px solid var(--border)', paddingTop: '14px', marginTop: '4px' }}>
      <button
        onClick={() => setOpen(o => !o)}
        className="btn btn-ghost"
        style={{ padding: '6px 12px', fontSize: '12px', color: 'var(--warning)', borderColor: 'var(--warning-border)' }}
      >
        <span style={{ transform: open ? 'rotate(90deg)' : 'none', transition: 'transform 0.2s', display: 'inline-block' }}>▶</span>
        How the contradiction was handled ({conflictPairs.length})
      </button>

      {open && (
        <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {confidenceReason && (
            <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{confidenceReason}</p>
          )}
          {conflictPairs.map((pair, i) => {
            const a = sourceMap.get(pair.chunk_id_a)
            const b = sourceMap.get(pair.chunk_id_b)
            return (
              <div key={i} style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                {[a, b].map((src, j) => src ? (
                  <div
                    key={j}
                    className="card"
                    style={{
                      flex: 1, minWidth: '180px',
                      borderColor: src.excluded_reason ? 'var(--danger-border)' : 'var(--success-border)',
                    }}
                  >
                    <div style={{ padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      <div style={{ fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px',
                                    color: src.excluded_reason ? 'var(--danger)' : 'var(--success)' }}>
                        {src.excluded_reason ? 'Set aside' : 'Preferred'}
                      </div>
                      <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)' }}>
                        {src.doc_title}{src.version_string ? ` v${src.version_string}` : ''}
                        {src.published_at ? ` · ${src.published_at}` : ''}
                      </div>
                      <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                        {src.snippet}
                      </div>
                    </div>
                  </div>
                ) : null)}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function QueryPage() {
  const { stats } = useShell()
  const [queryText,     setQueryText]     = useState('')
  const [maxChunks,     setMaxChunks]     = useState(20)
  const [retrieveOnly,  setRetrieveOnly]  = useState(false)
  const [compareMode,   setCompareMode]   = useState(false)
  const [loading,       setLoading]       = useState(false)
  const [result,        setResult]        = useState<QueryResponse | null>(null)
  const [compareResult, setCompareResult] = useState<{ standard: QueryResponse; temporal: QueryResponse } | null>(null)
  const [error,         setError]         = useState<string | null>(null)
  const [showSources,   setShowSources]   = useState(true)

  const corpusEmpty = stats !== null && stats.total_docs === 0

  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    try { return JSON.parse(localStorage.getItem('qp_history') || '[]') } catch { return [] }
  })

  const saveHistory = (q: string, r: QueryResponse) => {
    const entry: HistoryEntry = { query: q, latency_ms: r.latency_ms, conflicts: r.conflicts_detected ?? 0, ts: new Date().toISOString() }
    setHistory(prev => {
      const next = [entry, ...prev.filter(h => h.query !== q)].slice(0, 5)
      try { localStorage.setItem('qp_history', JSON.stringify(next)) } catch { /* quota — ignore */ }
      return next
    })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!queryText.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)
    setCompareResult(null)

    const payload = { query: queryText.trim(), max_chunks: maxChunks, retrieve_only: retrieveOnly }

    try {
      if (compareMode) {
        const [temporal, standard] = await Promise.all([
          axios.post<QueryResponse>('/api/v1/query', { ...payload, no_temporal: false }, { timeout: 90_000 }),
          axios.post<QueryResponse>('/api/v1/query', { ...payload, no_temporal: true  }, { timeout: 90_000 }),
        ])
        setCompareResult({ standard: standard.data, temporal: temporal.data })
        saveHistory(queryText.trim(), temporal.data)
      } else {
        const { data } = await axios.post<QueryResponse>('/api/v1/query', { ...payload, no_temporal: false }, { timeout: 60_000 })
        setResult(data)
        saveHistory(queryText.trim(), data)
      }
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        setError(err.response?.data?.detail ?? err.message)
      } else {
        setError('Unexpected error. Check backend logs.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Query</h1>
        <p className="page-subtitle">
          Ask a question. The system retrieves matching passages, prefers current
          versions over superseded ones, and flags sources that contradict each other.
        </p>
      </div>

      {corpusEmpty && (
        <div className="alert alert-info" style={{ marginBottom: '20px' }}>
          <div style={{ flex: 1 }}>
            <strong>No documents loaded yet.</strong> Add one on the Ingest page,
            then come back here to ask questions about it.
          </div>
          <a href="/ingest" className="btn btn-primary" style={{ fontSize: '12.5px', textDecoration: 'none', whiteSpace: 'nowrap' }}>
            Go to Ingest →
          </a>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '272px 1fr', gap: '24px', alignItems: 'start' }}>

        {/* ── Left panel: form + history ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', position: 'sticky', top: 0 }}>

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>

            <div className="field">
              <label htmlFor="query-input" className="field-label">Question</label>
              <textarea
                id="query-input"
                className="input"
                value={queryText}
                onChange={e => setQueryText(e.target.value)}
                placeholder="e.g. Is the old API still recommended?"
                rows={5}
                style={{ resize: 'vertical', lineHeight: 1.65 }}
              />
              <span className="field-hint">
                Mention a version (&ldquo;in v1.13&rdquo;) or use wording like
                &ldquo;previously&rdquo; to search superseded versions too.
              </span>
            </div>

            <div className="field">
              <label htmlFor="query-chunks" className="field-label">
                Passages to retrieve: <span style={{ color: 'var(--accent-hover)', fontWeight: 700 }}>{maxChunks}</span>
              </label>
              <input
                id="query-chunks"
                type="range" min={1} max={50} value={maxChunks}
                onChange={e => setMaxChunks(Number(e.target.value))}
                style={{ accentColor: 'var(--accent)', cursor: 'pointer' }}
              />
              <span className="field-hint">Only the highest-ranked ones fit in the answer.</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '12.5px', color: 'var(--text-secondary)' }}>
                <input
                  id="query-retrieve-only"
                  type="checkbox"
                  checked={retrieveOnly}
                  onChange={e => setRetrieveOnly(e.target.checked)}
                  style={{ accentColor: 'var(--accent)', width: '14px', height: '14px' }}
                />
                Retrieve only (skip the model)
              </label>

              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '12.5px', color: compareMode ? 'var(--warning)' : 'var(--text-secondary)' }}>
                <input
                  id="query-compare-toggle"
                  type="checkbox"
                  checked={compareMode}
                  onChange={e => setCompareMode(e.target.checked)}
                  style={{ accentColor: 'var(--warning)', width: '14px', height: '14px' }}
                />
                Compare against standard RAG
              </label>
            </div>

            {compareMode && (
              <div className="alert alert-warning" style={{ fontSize: '12px' }}>
                Runs the question twice — once with version-awareness and conflict
                handling, once without — and shows both answers side by side.
              </div>
            )}

            <button
              id="query-submit-btn"
              type="submit"
              className={`btn ${compareMode ? '' : 'btn-primary'}`}
              disabled={!queryText.trim() || loading}
              style={compareMode ? {
                background: 'linear-gradient(135deg, #d97706, #f59e0b)',
                color: '#fff',
                border: 'none',
              } : {}}
            >
              {loading ? (
                <>
                  <div className="spinner" style={{ width: '14px', height: '14px', borderColor: 'rgba(255,255,255,0.3)', borderTopColor: '#fff' }} />
                  {compareMode ? 'Comparing…' : 'Thinking…'}
                </>
              ) : compareMode ? 'Compare' : 'Ask'}
            </button>
          </form>

          {history.length > 0 && (
            <div className="card">
              <div className="card-header" style={{ padding: '10px 14px' }}>
                <span className="section-label">Recent questions</span>
              </div>
              <div style={{ padding: '4px 6px' }}>
                {history.map((h, i) => (
                  <button
                    key={i}
                    onClick={() => setQueryText(h.query)}
                    className="history-item"
                    title="Click to reuse this question"
                  >
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }} className="truncate">{h.query}</div>
                    <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', marginTop: '2px' }}>
                      {h.latency_ms} ms · {h.conflicts} conflict{h.conflicts !== 1 ? 's' : ''}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── Right panel: results ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', minWidth: 0 }}>

          {loading && (
            <div className="card">
              <div style={{ padding: '40px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '14px' }}>
                <div className="spinner spinner-lg" />
                <p style={{ color: 'var(--text-muted)', fontSize: '13px', textAlign: 'center' }}>
                  {retrieveOnly
                    ? 'Searching passages…'
                    : 'Searching → ranking by version and date → checking for contradictions → writing the answer…'}
                </p>
              </div>
            </div>
          )}

          {error && !loading && (
            <div className="alert alert-danger">
              <svg className="alert-icon" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-11a.75.75 0 00-1.5 0v4a.75.75 0 001.5 0V7zm-.75 7.5a.75.75 0 100-1.5.75.75 0 000 1.5z" clipRule="evenodd"/></svg>
              <div>
                <div style={{ fontWeight: 600, marginBottom: '3px' }}>Query failed</div>
                <div style={{ fontSize: '12.5px', opacity: 0.9 }}>{error}</div>
              </div>
            </div>
          )}

          {!result && !compareResult && !loading && !error && !corpusEmpty && (
            <div className="card">
              <div className="empty-state" style={{ padding: '56px 32px' }}>
                <div className="empty-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/>
                  </svg>
                </div>
                <div className="empty-title">Ask a question</div>
                <p className="empty-desc">
                  Every answer shows which passages it used, which version each came
                  from, and whether any of them disagreed with one another.
                </p>
              </div>
            </div>
          )}

          {/* ── Compare mode ── */}
          {compareResult && !loading && (
            <CompareView standard={compareResult.standard} temporal={compareResult.temporal} />
          )}

          {/* ── Single result ── */}
          {result && !loading && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>

              {/* How the question was read */}
              {result.analysis.source === 'default' && (
                <div className="alert alert-warning" style={{ fontSize: '12.5px' }}>
                  The question could not be interpreted ({result.analysis.error}), so it was
                  treated as a question about the <strong>current</strong> state.
                </div>
              )}
              {result.analysis.source === 'llm' && describeAnalysis(result.analysis) && (
                <div className="alert alert-info" style={{ fontSize: '12.5px' }}>
                  Read as <strong>{describeAnalysis(result.analysis)}</strong>.
                </div>
              )}
              {result.temporal_filter && (
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Only passages <strong>{result.temporal_filter.rule}</strong> were eligible:{' '}
                  {result.temporal_filter.candidates_valid} of {result.temporal_filter.candidates_retrieved}{' '}
                  retrieved passages passed, ranked by {result.temporal_filter.scoring}.
                </div>
              )}

              {result.conflicts_detected > 0 && (
                <div className="alert alert-warning">
                  <svg className="alert-icon" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd"/></svg>
                  <div style={{ flex: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px' }}>
                    <span>
                      <strong>{result.conflicts_detected}</strong> pair{result.conflicts_detected !== 1 ? 's' : ''} of
                      retrieved sources contradict each other. See how it was handled below.
                    </span>
                    <a href="/conflicts" style={{ color: 'var(--warning)', fontSize: '12px', fontWeight: 700, whiteSpace: 'nowrap' }}>
                      All conflicts →
                    </a>
                  </div>
                </div>
              )}

              <div style={{ display: 'flex', gap: '7px', flexWrap: 'wrap', alignItems: 'center' }}>
                <span className="badge badge-accent">{result.latency_ms} ms</span>
                <span className="badge badge-neutral">
                  {result.sources_used_in_answer} of {result.sources.length} passages used
                </span>
                {retrieveOnly && <span className="badge badge-warning">Retrieve-only</span>}
                {result.answer_confidence && result.answer_confidence !== 'none' && (
                  <ConfidenceBadge level={result.answer_confidence} reason={result.confidence_reason} />
                )}
              </div>

              {result.answer ? (
                <div className="card">
                  <div className="card-header">
                    <span className="card-title">Answer</span>
                  </div>
                  <div className="card-body">
                    <div className="md-body">
                      <ReactMarkdown>{result.answer}</ReactMarkdown>
                    </div>

                    {result.conflicts_detected > 0 && (
                      <ConflictDetailsPanel
                        sources={result.sources}
                        conflictPairs={result.conflict_pairs}
                        confidenceReason={result.confidence_reason}
                      />
                    )}
                  </div>
                </div>
              ) : (
                <div style={{ color: 'var(--text-muted)', fontSize: '13px', fontStyle: 'italic', padding: '12px 0' }}>
                  {result.temporal_filter && result.temporal_filter.candidates_valid === 0
                    ? <>No retrieved passage is {result.temporal_filter.rule}, so no answer was generated.</>
                    : retrieveOnly
                      ? 'No answer generated (retrieve-only mode).'
                      : 'No relevant passages were found, so no answer was generated.'}
                </div>
              )}

              {result.sources.length > 0 ? (
                <div>
                  <button
                    onClick={() => setShowSources(s => !s)}
                    className="btn btn-ghost"
                    style={{ padding: '6px 12px', fontSize: '12.5px', marginBottom: '10px' }}
                  >
                    {showSources ? '▼' : '▶'} Sources ({result.sources.length})
                  </button>

                  {showSources && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {result.sources.map(src => (
                        <SourceCard
                          key={src.chunk_id}
                          source={src}
                          temporalApplied={result.temporal_pipeline_applied}
                        />
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="alert alert-info" style={{ fontSize: '12.5px' }}>
                  No passages matched this question. Try different wording, or check
                  the Library to see what is loaded.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Compare view ───────────────────────────────────────────────────────────

function CompareView({ standard, temporal }: { standard: QueryResponse; temporal: QueryResponse }) {
  const columns: Array<{ key: string; label: string; hint: string; data: QueryResponse; accent: boolean }> = [
    {
      key: 'standard',
      label: 'Standard RAG',
      hint: 'Relevance only — no version awareness, no conflict handling.',
      data: standard,
      accent: false,
    },
    {
      key: 'temporal',
      label: 'Temporal RAG',
      hint: 'Prefers current versions and reconciles contradictions.',
      data: temporal,
      accent: true,
    },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      <div>
        <div className="section-label">Side-by-side comparison</div>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
          The same question answered with and without the temporal pipeline.
        </p>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
        {columns.map(({ key, label, hint, data, accent }) => (
          <div key={key} style={{ display: 'flex', flexDirection: 'column', gap: '10px', minWidth: 0 }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                <span className={`badge ${accent ? 'badge-accent' : 'badge-neutral'}`}>{label}</span>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{data.latency_ms} ms</span>
                {accent && data.answer_confidence && data.answer_confidence !== 'none' && (
                  <ConfidenceBadge level={data.answer_confidence} reason={data.confidence_reason} />
                )}
                {accent && data.conflicts_detected > 0 && (
                  <span className="badge badge-warning">{data.conflicts_detected} conflict{data.conflicts_detected !== 1 ? 's' : ''}</span>
                )}
              </div>
              <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '5px' }}>{hint}</p>
            </div>

            <div className="card" style={accent ? { borderColor: 'var(--accent-border)' } : undefined}>
              <div className="card-body" style={{ fontSize: '13px', lineHeight: 1.75 }}>
                <div className="md-body">
                  <ReactMarkdown>{data.answer ?? '*No answer generated*'}</ReactMarkdown>
                </div>
              </div>
            </div>

            <div className="section-label">Top sources</div>
            {data.sources.slice(0, 4).map((s, i) => (
              <div key={s.chunk_id} className="card">
                <div style={{ padding: '10px 12px' }}>
                  <div style={{ fontWeight: 600, fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                    {i + 1}. {s.doc_title}{s.version_string ? ` v${s.version_string}` : ''}
                    {s.is_superseded
                      ? <span style={{ color: 'var(--warning)', fontSize: '10px', fontWeight: 700 }}>SUPERSEDED</span>
                      : s.is_latest
                        ? <span style={{ color: 'var(--success)', fontSize: '10px', fontWeight: 700 }}>CURRENT</span>
                        : null}
                  </div>
                  <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                    {s.snippet.slice(0, 120)}…
                  </div>
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
