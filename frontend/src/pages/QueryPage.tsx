import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import axios from 'axios'

// ── Types ──────────────────────────────────────────────────────────────────

interface SourceResult {
  chunk_id: string
  doc_title: string
  snippet: string
  bm25_score: number
  semantic_score: number
  rrf_score: number
  version_string?: string | null
  published_at?: string | null
  is_latest?: boolean | null
  temporal_score?: number | null
  composite_score?: number | null
  has_conflict?: boolean | null
}

interface QueryResponse {
  query_id: string
  answer: string | null
  latency_ms: number
  sources: SourceResult[]
}

// ── Sub-components ─────────────────────────────────────────────────────────

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.min(100, Math.round(value * 100))
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px' }}>
        <span style={{ color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.4px' }}>{label}</span>
        <span style={{ color, fontWeight: 600 }}>{value.toFixed(3)}</span>
      </div>
      <div style={{ height: '4px', background: 'var(--bg-tertiary)', borderRadius: '2px', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: '2px', transition: 'width 0.4s ease' }} />
      </div>
    </div>
  )
}

function SourceCard({ source, index }: { source: SourceResult; index: number }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-md)',
      padding: '14px',
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
      transition: 'border-color 0.2s',
    }}
      onMouseEnter={e => (e.currentTarget.style.borderColor = 'rgba(99,102,241,0.5)')}
      onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border)')}
    >
      {/* Header */}
      <div style={{ fontWeight: 600, fontSize: '12px', color: 'var(--text-primary)', lineHeight: 1.4 }}>
        [{index}] {source.doc_title}
      </div>

      {/* Snippet */}
      <div
        onClick={() => setExpanded(e => !e)}
        style={{
          fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.6,
          cursor: 'pointer', display: '-webkit-box',
          WebkitLineClamp: expanded ? 'unset' : 2,
          WebkitBoxOrient: 'vertical', overflow: 'hidden',
        }}
        title="Click to expand"
      >
        {source.snippet}
      </div>

      {/* Score bars */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        <ScoreBar label="BM25"     value={source.bm25_score}     color="#f59e0b" />
        <ScoreBar label="Semantic" value={source.semantic_score}  color="#06b6d4" />
        <ScoreBar label="RRF"      value={source.rrf_score * 60}  color="#6366f1" />
      </div>

      {/* Future phase badges (hidden until populated) */}
      {source.is_latest !== null && source.is_latest !== undefined && (
        <div style={{ fontSize: '10px', color: source.is_latest ? 'var(--accent-success)' : '#f59e0b', fontWeight: 600 }}>
          {source.is_latest ? '✓ Latest version' : '⚠ Older version'}
        </div>
      )}
      {source.has_conflict && (
        <div style={{ fontSize: '10px', color: '#ef4444', fontWeight: 600 }}>
          ⚡ Conflict detected
        </div>
      )}
    </div>
  )
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function QueryPage() {
  const [queryText, setQueryText] = useState('')
  const [maxChunks, setMaxChunks] = useState(20)
  const [retrieveOnly, setRetrieveOnly] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!queryText.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const { data } = await axios.post<QueryResponse>('/api/v1/query', {
        query: queryText.trim(),
        max_chunks: maxChunks,
        retrieve_only: retrieveOnly,
      }, { timeout: 60_000 })
      setResult(data)
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
        <p className="page-subtitle">Ask a question — the system retrieves relevant chunks and generates an answer via Groq.</p>
      </div>

      {/* ── Three-column layout ─────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr 280px', gap: '20px', alignItems: 'start' }}>

        {/* ── LEFT: Query form ─────────────────────────────────────────── */}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label htmlFor="query-input" style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Question
            </label>
            <textarea
              id="query-input"
              value={queryText}
              onChange={e => setQueryText(e.target.value)}
              placeholder="How does torch.autograd work?"
              rows={5}
              style={{
                background: 'var(--bg-card)', border: '1px solid var(--border-light)',
                borderRadius: 'var(--radius-sm)', padding: '10px 12px',
                color: 'var(--text-primary)', fontSize: '13px', resize: 'vertical',
                outline: 'none', fontFamily: 'inherit', lineHeight: 1.6,
                transition: 'border-color 0.2s',
              }}
              onFocus={e => (e.target.style.borderColor = 'var(--accent-primary)')}
              onBlur={e => (e.target.style.borderColor = 'var(--border-light)')}
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label htmlFor="query-chunks" style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Max Sources: <span style={{ color: 'var(--accent-primary-hover)', fontWeight: 700 }}>{maxChunks}</span>
            </label>
            <input
              id="query-chunks"
              type="range" min={1} max={50} value={maxChunks}
              onChange={e => setMaxChunks(Number(e.target.value))}
              style={{ accentColor: 'var(--accent-primary)' }}
            />
          </div>

          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '12px', color: 'var(--text-secondary)' }}>
            <input
              id="query-retrieve-only"
              type="checkbox"
              checked={retrieveOnly}
              onChange={e => setRetrieveOnly(e.target.checked)}
              style={{ accentColor: 'var(--accent-primary)', width: '14px', height: '14px' }}
            />
            Retrieve only (no Groq call)
          </label>

          <button
            id="query-submit-btn"
            type="submit"
            disabled={!queryText.trim() || loading}
            style={{
              background: !queryText.trim() || loading ? 'var(--bg-hover)' : 'linear-gradient(135deg, var(--accent-primary), #4f46e5)',
              border: 'none', borderRadius: 'var(--radius-sm)', padding: '11px 20px',
              color: !queryText.trim() || loading ? 'var(--text-muted)' : 'white',
              fontSize: '14px', fontWeight: 600,
              cursor: !queryText.trim() || loading ? 'not-allowed' : 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
            }}
          >
            {loading ? (
              <>
                <span style={{ display: 'inline-block', width: '14px', height: '14px', border: '2px solid rgba(255,255,255,0.3)', borderTopColor: 'white', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />
                Thinking…
              </>
            ) : '⚡ Ask Groq'}
          </button>
        </form>

        {/* ── CENTER: Answer panel ──────────────────────────────────────── */}
        <div style={{ minHeight: '300px' }}>
          {loading && (
            <div style={{
              background: 'var(--bg-card)', border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)', padding: '32px',
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px',
            }}>
              <div style={{ width: '36px', height: '36px', border: '3px solid var(--border)', borderTopColor: 'var(--accent-primary)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
              <p style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
                {retrieveOnly ? 'Retrieving chunks…' : 'Retrieving → building context → asking Groq…'}
              </p>
            </div>
          )}

          {error && !loading && (
            <div style={{ background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.3)', borderRadius: 'var(--radius-md)', padding: '20px', display: 'flex', gap: '10px' }}>
              <span style={{ fontSize: '18px' }}>⚠</span>
              <div>
                <div style={{ fontWeight: 600, color: 'var(--accent-error)', marginBottom: '4px' }}>Query failed</div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>{error}</div>
              </div>
            </div>
          )}

          {result && !loading && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              {/* Latency + meta */}
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <span style={{ background: 'rgba(99,102,241,0.12)', color: 'var(--accent-primary-hover)', borderRadius: '999px', padding: '3px 10px', fontSize: '11px', fontWeight: 600 }}>
                  ⏱ {result.latency_ms} ms
                </span>
                <span style={{ background: 'rgba(52,211,153,0.1)', color: 'var(--accent-success)', borderRadius: '999px', padding: '3px 10px', fontSize: '11px', fontWeight: 600 }}>
                  {result.sources.length} source{result.sources.length !== 1 ? 's' : ''}
                </span>
                {retrieveOnly && (
                  <span style={{ background: 'rgba(245,158,11,0.12)', color: '#f59e0b', borderRadius: '999px', padding: '3px 10px', fontSize: '11px', fontWeight: 600 }}>
                    Retrieve-only
                  </span>
                )}
              </div>

              {/* Answer */}
              {result.answer ? (
                <div style={{
                  background: 'var(--bg-card)', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)', padding: '20px 24px',
                }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '12px' }}>
                    Answer
                  </div>
                  <div style={{ color: 'var(--text-primary)', fontSize: '14px', lineHeight: 1.8 }} className="markdown-body">
                    <ReactMarkdown>{result.answer}</ReactMarkdown>
                  </div>
                </div>
              ) : (
                <div style={{ color: 'var(--text-muted)', fontSize: '13px', fontStyle: 'italic', padding: '12px 0' }}>
                  No answer generated (retrieve-only mode).
                </div>
              )}
            </div>
          )}

          {!result && !loading && !error && (
            <div style={{
              background: 'var(--bg-card)', border: '1px dashed var(--border)',
              borderRadius: 'var(--radius-md)', padding: '40px',
              textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px',
            }}>
              <div style={{ fontSize: '32px', marginBottom: '12px' }}>💬</div>
              Ask a question to see the answer here.
            </div>
          )}
        </div>

        {/* ── RIGHT: Source cards ───────────────────────────────────────── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {result && result.sources.length > 0 && (
            <>
              <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Sources
              </div>
              {result.sources.map((src, i) => (
                <SourceCard key={src.chunk_id} source={src} index={i + 1} />
              ))}
            </>
          )}
          {result && result.sources.length === 0 && (
            <div style={{ color: 'var(--text-muted)', fontSize: '13px', padding: '12px 0' }}>
              No relevant sources found. Try ingesting documents first.
            </div>
          )}
          {!result && (
            <div style={{ color: 'var(--text-muted)', fontSize: '12px', lineHeight: 1.8, paddingTop: '4px' }}>
              <strong style={{ color: 'var(--text-secondary)', display: 'block', marginBottom: '8px' }}>Score bars explained</strong>
              <div>🟡 BM25 — keyword match</div>
              <div>🔵 Semantic — embedding similarity</div>
              <div>🟣 RRF — fused rank (used for ordering)</div>
            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        .markdown-body h1, .markdown-body h2, .markdown-body h3 {
          color: var(--text-primary);
          margin: 16px 0 8px;
          font-weight: 600;
        }
        .markdown-body p { margin: 0 0 10px; }
        .markdown-body code {
          background: var(--bg-tertiary);
          padding: 1px 5px;
          border-radius: 3px;
          font-size: 12px;
          font-family: 'Fira Code', monospace;
        }
        .markdown-body pre {
          background: var(--bg-tertiary);
          border: 1px solid var(--border);
          border-radius: var(--radius-sm);
          padding: 12px;
          overflow-x: auto;
          font-size: 12px;
        }
        .markdown-body ul, .markdown-body ol { padding-left: 20px; margin: 8px 0; }
        .markdown-body li { margin: 4px 0; }
        .markdown-body strong { color: var(--text-primary); font-weight: 600; }
      `}</style>
    </div>
  )
}
