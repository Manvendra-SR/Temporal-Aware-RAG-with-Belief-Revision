import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

// ── Types ───────────────────────────────────────────────────────────────────

interface ChunkInfo {
  chunk_id: string
  snippet: string
  section_heading: string | null
  doc_title: string
  version_string: string | null
  published_at: string | null
  is_latest: boolean
}

interface ConflictRecord {
  conflict_id: string
  conflict_type: string
  nli_score: number
  detected_at: string
  detected_during: string | null
  is_resolved: boolean
  resolution_type: string | null
  chunk_a: ChunkInfo
  chunk_b: ChunkInfo
}

interface ConflictsResponse {
  total: number
  page: number
  limit: number
  conflicts: ConflictRecord[]
  total_unresolved: number
  total_resolved: number
}

// ── Helpers ─────────────────────────────────────────────────────────────────

// The detector only ever emits these two types (see services/conflict_detector
// _classify). "scope_change" was offered as a filter option but nothing could
// ever produce it, so the filter silently returned an empty list.
const CONFLICT_TYPES = [
  {
    value: 'version_supersession',
    label: 'Newer version disagrees',
    help: 'The sources contradict each other and are far enough apart in time that one clearly supersedes the other.',
  },
  {
    value: 'direct_contradiction',
    label: 'Direct contradiction',
    help: 'The sources make conflicting claims without a clear time ordering between them.',
  },
] as const

function typeLabel(t: string) {
  return CONFLICT_TYPES.find(x => x.value === t)?.label ?? t
}

function typeHelp(t: string) {
  return CONFLICT_TYPES.find(x => x.value === t)?.help ?? ''
}

function typeClass(t: string) {
  return t === 'direct_contradiction' ? 'badge-danger'
       : t === 'version_supersession' ? 'badge-warning'
       : 'badge-accent'
}

function nliClass(score: number) {
  if (score >= 0.85) return 'badge-danger'
  if (score >= 0.70) return 'badge-warning'
  return 'badge-accent'
}

function fmtDate(iso: string) {
  try { return new Date(iso).toLocaleString() } catch { return iso }
}

/** Order a conflicting pair oldest-first. Undated sides keep their original order. */
function orderByDate(a: ChunkInfo, b: ChunkInfo): [ChunkInfo, ChunkInfo] {
  if (!a.published_at || !b.published_at) return [a, b]
  return a.published_at <= b.published_at ? [a, b] : [b, a]
}

/** Summarise which two things disagree, by version when the titles match. */
function describePair(record: ConflictRecord): string {
  const [older, newer] = orderByDate(record.chunk_a, record.chunk_b)
  const v = (c: ChunkInfo) => (c.version_string ? `v${c.version_string}` : 'unversioned')

  if (older.doc_title === newer.doc_title) {
    return `${v(older)} → ${v(newer)}`
  }
  return `${older.doc_title} ${v(older)} vs. ${newer.doc_title} ${v(newer)}`
}

// ── Chunk Side ───────────────────────────────────────────────────────────────

function ChunkSide({ info, label }: { info: ChunkInfo; label: string }) {
  return (
    <div
      className="card"
      style={{ flex: 1, minWidth: 0 }}
    >
      <div style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          {label}
        </div>
        <div style={{ fontWeight: 600, fontSize: '12.5px', color: 'var(--text-primary)' }}>
          {info.doc_title}
        </div>
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', alignItems: 'center' }}>
          {info.version_string && (
            <span
              className="badge"
              style={{
                background: info.is_latest ? 'var(--success-subtle)' : 'var(--warning-subtle)',
                color: info.is_latest ? 'var(--success)' : 'var(--warning)',
                border: `1px solid ${info.is_latest ? 'var(--success-border)' : 'var(--warning-border)'}`,
                fontFamily: 'ui-monospace, monospace',
              }}
            >
              v{info.version_string}
            </span>
          )}
          {info.published_at && (
            <span style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>📅 {info.published_at}</span>
          )}
          {info.is_latest && (
            <span style={{ fontSize: '10.5px', color: 'var(--success)', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '3px' }}>
              <span className="dot dot-success" />Latest
            </span>
          )}
        </div>
        {info.section_heading && (
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
            § {info.section_heading}
          </div>
        )}
        <div style={{
          fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.65,
          background: 'var(--bg-elevated)',
          borderRadius: 'var(--radius-md)',
          padding: '10px 12px',
          border: '1px solid var(--border)',
        }}>
          {info.snippet}
        </div>
      </div>
    </div>
  )
}

// ── Conflict Row ─────────────────────────────────────────────────────────────

function ConflictRow({
  record: initialRecord,
  onResolved,
}: {
  record: ConflictRecord
  onResolved?: (updated: ConflictRecord) => void
}) {
  const [record,         setRecord]         = useState<ConflictRecord>(initialRecord)
  const [expanded,       setExpanded]       = useState(false)
  const [resolveOpen,    setResolveOpen]    = useState(false)
  const [resolutionType, setResolutionType] = useState('temporal_preference')
  const [resolutionNote, setResolutionNote] = useState('')
  const [submitting,     setSubmitting]     = useState(false)
  const [resolveError,   setResolveError]   = useState<string | null>(null)

  const handleResolve = async () => {
    setSubmitting(true)
    setResolveError(null)
    try {
      const { data } = await axios.post<ConflictRecord>(
        `/api/v1/conflicts/${record.conflict_id}/resolve`,
        { resolution_type: resolutionType, resolution_note: resolutionNote },
      )
      setRecord(data)
      setResolveOpen(false)
      onResolved?.(data)
    } catch (err: unknown) {
      setResolveError(axios.isAxiosError(err) ? (err.response?.data?.detail ?? err.message) : 'Unexpected error')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div
      className="card"
      style={{ transition: 'border-color 0.15s' }}
      onMouseEnter={e => { if (!expanded) (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-focus)' }}
      onMouseLeave={e => { if (!expanded) (e.currentTarget as HTMLElement).style.borderColor = 'var(--border)' }}
    >
      {/* Summary row */}
      <div
        onClick={() => setExpanded(v => !v)}
        style={{
          padding: '14px 18px',
          display: 'grid',
          gridTemplateColumns: '1fr auto auto auto auto',
          gap: '12px',
          alignItems: 'center',
          cursor: 'pointer',
        }}
      >
        {/* Docs involved. Versions are shown because both sides are usually
            two versions of the SAME document, where the title alone reads as
            a meaningless "X vs. X". */}
        <div style={{ minWidth: 0 }}>
          <div style={{ fontWeight: 600, fontSize: '13px', color: 'var(--text-primary)' }} className="truncate">
            {record.chunk_a.doc_title}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }} className="truncate">
            {describePair(record)}
          </div>
        </div>

        {/* Type badge */}
        <span
          className={`badge ${typeClass(record.conflict_type)}`}
          title={typeHelp(record.conflict_type)}
        >
          {typeLabel(record.conflict_type)}
        </span>

        {/* Contradiction score. Labelled in plain language — "NLI" is an
            implementation detail that means nothing to a reader. */}
        <span
          className={`badge ${nliClass(record.nli_score)}`}
          style={{ fontFamily: 'ui-monospace, monospace' }}
          title="How confident the contradiction model is that these two passages disagree."
        >
          {/* One decimal place: scores routinely sit at 0.997, and rounding
              those to a flat "100%" claims a certainty the model never states. */}
          {(record.nli_score * 100).toFixed(1)}% confident
        </span>

        {/* Status */}
        {record.is_resolved ? (
          <span className="badge badge-success">Resolved</span>
        ) : (
          <span className="badge badge-warning">Needs review</span>
        )}

        {/* Chevron */}
        <span style={{
          color: 'var(--text-muted)', fontSize: '11px',
          transform: expanded ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.2s',
        }}>▾</span>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div style={{ borderTop: '1px solid var(--border)', padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: '14px' }}>

          {/* Meta row */}
          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', fontSize: '12px', color: 'var(--text-muted)' }}>
            <span>Detected: <strong style={{ color: 'var(--text-secondary)' }}>{fmtDate(record.detected_at)}</strong></span>
            {record.detected_during && (
              <span>During query: <code style={{ fontSize: '11px', color: 'var(--text-secondary)', background: 'var(--bg-elevated)', padding: '1px 5px', borderRadius: 'var(--radius-sm)' }}>
                {record.detected_during.slice(0, 8)}…
              </code></span>
            )}
          </div>

          {/* Side-by-side, ordered oldest → newest so the reader can see at a
              glance which claim the system treats as superseded. */}
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            {orderByDate(record.chunk_a, record.chunk_b).map((info, i) => (
              <ChunkSide
                key={info.chunk_id}
                info={info}
                label={i === 0 ? 'Earlier claim' : 'Later claim'}
              />
            ))}
          </div>

          {/* Resolve section */}
          {!record.is_resolved ? (
            <div>
              {!resolveOpen ? (
                <button
                  onClick={() => setResolveOpen(true)}
                  className="btn btn-secondary"
                  style={{ fontSize: '12.5px', padding: '7px 16px' }}
                >
                  Mark as resolved
                </button>
              ) : (
                <div
                  className="card"
                  style={{ background: 'var(--bg-elevated)' }}
                >
                  <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>Resolve this conflict</div>

                    <div className="field">
                      <label className="field-label">Resolution type</label>
                      <select
                        className="input"
                        value={resolutionType}
                        onChange={e => setResolutionType(e.target.value)}
                      >
                        <option value="temporal_preference">Temporal preference (newer wins)</option>
                        <option value="manual">Manual resolution</option>
                        <option value="scope_clarification">Scope clarification</option>
                      </select>
                    </div>

                    <div className="field">
                      <label className="field-label">Note (optional)</label>
                      <textarea
                        className="input"
                        value={resolutionNote}
                        onChange={e => setResolutionNote(e.target.value)}
                        placeholder="Why was this resolved this way?"
                        rows={2}
                      />
                    </div>

                    {resolveError && (
                      <div className="alert alert-danger" style={{ fontSize: '12px' }}>
                        {resolveError}
                      </div>
                    )}

                    <div style={{ display: 'flex', gap: '8px' }}>
                      <button
                        onClick={handleResolve}
                        disabled={submitting}
                        className="btn btn-primary"
                        style={{ fontSize: '12.5px', padding: '7px 16px' }}
                      >
                        {submitting ? (
                          <>
                            <div className="spinner" style={{ width: '13px', height: '13px', borderColor: 'rgba(255,255,255,0.3)', borderTopColor: '#fff' }} />
                            Saving…
                          </>
                        ) : 'Submit resolution'}
                      </button>
                      <button
                        onClick={() => { setResolveOpen(false); setResolveError(null) }}
                        className="btn btn-ghost"
                        style={{ fontSize: '12.5px' }}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12.5px', color: 'var(--success)', fontWeight: 600 }}>
              <span className="dot dot-success" />
              Resolved — {record.resolution_type}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Main Page ────────────────────────────────────────────────────────────────

const LIMIT = 20

export default function ConflictsPage() {
  const [data,           setData]           = useState<ConflictsResponse | null>(null)
  const [loading,        setLoading]        = useState(false)
  const [error,          setError]          = useState<string | null>(null)
  const [page,           setPage]           = useState(1)
  const [filterType,     setFilterType]     = useState<string>('')
  const [filterResolved, setFilterResolved] = useState<string>('')

  const fetchConflicts = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const params: Record<string, string | number> = { page, limit: LIMIT }
      if (filterType)       params['type']        = filterType
      if (filterResolved !== '') params['is_resolved'] = filterResolved
      const { data: res } = await axios.get<ConflictsResponse>('/api/v1/conflicts', { params })
      setData(res)
    } catch (err: unknown) {
      setError(axios.isAxiosError(err) ? (err.response?.data?.detail ?? err.message) : 'Unexpected error')
    } finally {
      setLoading(false)
    }
  }, [page, filterType, filterResolved])

  useEffect(() => { fetchConflicts() }, [fetchConflicts])

  const totalPages = data ? Math.ceil(data.total / LIMIT) : 1

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Conflicts</h1>
        <p className="page-subtitle">
          Passages that contradict each other, found while answering questions.
          Expand one to compare both sides and record how it should be resolved.
        </p>
      </div>

      {/* Corpus-wide counts. These come from the server rather than being
          counted off the current page, which previously made them disagree
          with the total as soon as there was more than one page. */}
      {data && (
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '24px' }}>
          {[
            { label: 'Total found', value: data.total_unresolved + data.total_resolved },
            { label: 'Needs review', value: data.total_unresolved },
            { label: 'Resolved',    value: data.total_resolved },
          ].map(({ label, value }) => (
            <div
              key={label}
              className="card"
              style={{ padding: '12px 20px', display: 'flex', flexDirection: 'column', gap: '2px', minWidth: '100px' }}
            >
              <div style={{ fontSize: '22px', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.5px' }}>{value}</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.4px', fontWeight: 600 }}>{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Filter bar */}
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '20px', alignItems: 'center' }}>
        <select
          className="input"
          value={filterType}
          onChange={e => { setFilterType(e.target.value); setPage(1) }}
          style={{ width: 'auto', padding: '7px 32px 7px 12px' }}
        >
          <option value="">All types</option>
          {CONFLICT_TYPES.map(t => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>

        <select
          className="input"
          value={filterResolved}
          onChange={e => { setFilterResolved(e.target.value); setPage(1) }}
          style={{ width: 'auto', padding: '7px 32px 7px 12px' }}
        >
          <option value="">All statuses</option>
          <option value="false">Needs review</option>
          <option value="true">Resolved</option>
        </select>

        <button onClick={fetchConflicts} className="btn btn-secondary" style={{ fontSize: '12.5px', padding: '7px 14px' }}>
          ↺ Refresh
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <div className="loading-center"><div className="spinner spinner-lg" /></div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="alert alert-danger">
          <svg className="alert-icon" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-11a.75.75 0 00-1.5 0v4a.75.75 0 001.5 0V7zm-.75 7.5a.75.75 0 100-1.5.75.75 0 000 1.5z" clipRule="evenodd"/></svg>
          {error}
        </div>
      )}

      {/* Empty state */}
      {data && !loading && data.conflicts.length === 0 && (
        <div className="card">
          <div className="empty-state">
            <div className="empty-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>
            </div>
            <div className="empty-title">
              {filterType || filterResolved ? 'No conflicts match' : 'No conflicts detected'}
            </div>
            <p className="empty-desc">
              {filterType || filterResolved
                ? 'Try changing the filters above.'
                : 'Run queries to detect contradictions between document versions.'}
            </p>
          </div>
        </div>
      )}

      {/* Conflict list */}
      {data && !loading && data.conflicts.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {data.conflicts.map(r => (
            <ConflictRow
              key={r.conflict_id}
              record={r}
              onResolved={() => fetchConflicts()}
            />
          ))}
        </div>
      )}

      {/* Pagination */}
      {data && totalPages > 1 && (
        <div className="pagination">
          <button
            className="btn btn-ghost"
            style={{ padding: '6px 14px', fontSize: '12.5px' }}
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page <= 1}
          >
            ← Prev
          </button>
          <span className="pagination-info">{page} / {totalPages}</span>
          <button
            className="btn btn-ghost"
            style={{ padding: '6px 14px', fontSize: '12.5px' }}
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
          >
            Next →
          </button>
        </div>
      )}

    </div>
  )
}
