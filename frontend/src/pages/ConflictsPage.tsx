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
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function typeLabel(t: string) {
  return t === 'direct_contradiction' ? '⚡ Direct Contradiction'
    : t === 'version_supersession'    ? '🔄 Version Supersession'
    : t === 'scope_change'            ? '📐 Scope Change'
    : t
}

function typeColor(t: string) {
  return t === 'direct_contradiction' ? '#ef4444'
    : t === 'version_supersession'    ? '#f59e0b'
    : '#6366f1'
}

function nliColor(score: number) {
  if (score >= 0.85) return '#ef4444'
  if (score >= 0.70) return '#f59e0b'
  return '#6366f1'
}

function fmtDate(iso: string) {
  try { return new Date(iso).toLocaleString() } catch { return iso }
}

// ── Sub-components ──────────────────────────────────────────────────────────

function ChunkSide({ info, label }: { info: ChunkInfo; label: string }) {
  return (
    <div style={{
      flex: 1, minWidth: 0,
      background: 'var(--bg-secondary)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-sm)',
      padding: '12px',
      display: 'flex', flexDirection: 'column', gap: '8px',
    }}>
      <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.4px' }}>
        {label}
      </div>
      <div style={{ fontWeight: 600, fontSize: '12px', color: 'var(--text-primary)' }}>
        {info.doc_title}
      </div>
      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
        {info.version_string && (
          <span style={{
            background: info.is_latest ? 'rgba(52,211,153,0.15)' : 'rgba(245,158,11,0.12)',
            color: info.is_latest ? 'var(--accent-success)' : '#f59e0b',
            borderRadius: '999px', padding: '2px 8px', fontSize: '10px', fontWeight: 700,
          }}>v{info.version_string}</span>
        )}
        {info.published_at && (
          <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>📅 {info.published_at}</span>
        )}
        {info.is_latest && (
          <span style={{ fontSize: '10px', color: 'var(--accent-success)', fontWeight: 700 }}>● Latest</span>
        )}
      </div>
      {info.section_heading && (
        <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          § {info.section_heading}
        </div>
      )}
      <div style={{
        fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.6,
        background: 'var(--bg-card)', borderRadius: '4px', padding: '8px',
        border: '1px solid var(--border)',
      }}>
        {info.snippet}
      </div>
    </div>
  )
}

function ConflictRow({
  record: initialRecord,
  onResolved,
}: {
  record: ConflictRecord
  onResolved?: (updated: ConflictRecord) => void
}) {
  const [record, setRecord] = useState<ConflictRecord>(initialRecord)
  const [expanded, setExpanded] = useState(false)
  const [resolveOpen, setResolveOpen] = useState(false)
  const [resolutionType, setResolutionType] = useState('temporal_preference')
  const [resolutionNote, setResolutionNote] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [resolveError, setResolveError] = useState<string | null>(null)

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
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-md)',
      overflow: 'hidden',
      transition: 'border-color 0.2s',
    }}
      onMouseEnter={e => (e.currentTarget.style.borderColor = 'rgba(99,102,241,0.4)')}
      onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border)')}
    >
      {/* Summary row */}
      <div
        onClick={() => setExpanded(v => !v)}
        style={{
          padding: '14px 16px',
          display: 'grid',
          gridTemplateColumns: '1fr auto auto auto auto',
          gap: '12px',
          alignItems: 'center',
          cursor: 'pointer',
        }}
      >
        {/* Docs involved */}
        <div style={{ minWidth: 0 }}>
          <div style={{ fontWeight: 600, fontSize: '12px', color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {record.chunk_a.doc_title}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            vs {record.chunk_b.doc_title}
          </div>
        </div>

        {/* Type badge */}
        <span style={{
          background: `${typeColor(record.conflict_type)}22`,
          color: typeColor(record.conflict_type),
          borderRadius: '999px', padding: '3px 10px',
          fontSize: '10px', fontWeight: 700, whiteSpace: 'nowrap',
        }}>
          {typeLabel(record.conflict_type)}
        </span>

        {/* NLI score */}
        <span style={{
          background: `${nliColor(record.nli_score)}22`,
          color: nliColor(record.nli_score),
          borderRadius: '999px', padding: '3px 10px',
          fontSize: '10px', fontWeight: 700, whiteSpace: 'nowrap',
        }}>
          NLI {record.nli_score.toFixed(3)}
        </span>

        {/* Resolved status */}
        <span style={{
          background: record.is_resolved ? 'rgba(52,211,153,0.1)' : 'rgba(245,158,11,0.1)',
          color: record.is_resolved ? 'var(--accent-success)' : '#f59e0b',
          borderRadius: '999px', padding: '3px 10px',
          fontSize: '10px', fontWeight: 700,
        }}>
          {record.is_resolved ? '✓ Resolved' : 'Pending'}
        </span>

        {/* Chevron */}
        <span style={{
          color: 'var(--text-muted)', fontSize: '12px',
          transform: expanded ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.2s',
        }}>▾</span>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div style={{ borderTop: '1px solid var(--border)', padding: '16px' }}>
          {/* Meta */}
          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', marginBottom: '16px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              🕐 Detected: <strong style={{ color: 'var(--text-secondary)' }}>{fmtDate(record.detected_at)}</strong>
            </span>
            {record.detected_during && (
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                🔗 Query: <code style={{ color: 'var(--text-secondary)', fontSize: '10px' }}>{record.detected_during.slice(0, 8)}…</code>
              </span>
            )}
          </div>

          {/* Side-by-side chunks */}
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <ChunkSide info={record.chunk_a} label="Source A" />
            <ChunkSide info={record.chunk_b} label="Source B" />
          </div>

          {/* Resolve section */}
          {!record.is_resolved ? (
            <div style={{ marginTop: '14px' }}>
              {!resolveOpen ? (
                <button
                  onClick={() => setResolveOpen(true)}
                  style={{
                    background: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.3)',
                    borderRadius: 'var(--radius-sm)', padding: '7px 16px',
                    color: 'var(--accent-primary-hover)', fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                  }}
                >
                  Resolve conflict
                </button>
              ) : (
                <div style={{
                  background: 'var(--bg-secondary)', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)', padding: '14px',
                  display: 'flex', flexDirection: 'column', gap: '10px',
                }}>
                  <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)' }}>Resolve this conflict</div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <label style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600 }}>Resolution type</label>
                    <select
                      value={resolutionType}
                      onChange={e => setResolutionType(e.target.value)}
                      style={{
                        background: 'var(--bg-card)', border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-sm)', padding: '6px 10px',
                        color: 'var(--text-primary)', fontSize: '12px', cursor: 'pointer',
                      }}
                    >
                      <option value="temporal_preference">Temporal preference (newer wins)</option>
                      <option value="manual">Manual resolution</option>
                      <option value="scope_clarification">Scope clarification</option>
                    </select>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <label style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600 }}>Note (optional)</label>
                    <textarea
                      value={resolutionNote}
                      onChange={e => setResolutionNote(e.target.value)}
                      placeholder="Why was this resolved this way?"
                      rows={2}
                      style={{
                        background: 'var(--bg-card)', border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-sm)', padding: '8px 10px',
                        color: 'var(--text-primary)', fontSize: '12px', resize: 'vertical',
                        fontFamily: 'inherit',
                      }}
                    />
                  </div>

                  {resolveError && (
                    <div style={{ fontSize: '11px', color: 'var(--accent-error)' }}>⚠ {resolveError}</div>
                  )}

                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                      onClick={handleResolve}
                      disabled={submitting}
                      style={{
                        background: 'var(--accent-primary)', border: 'none',
                        borderRadius: 'var(--radius-sm)', padding: '7px 16px',
                        color: '#fff', fontSize: '12px', fontWeight: 600,
                        cursor: submitting ? 'not-allowed' : 'pointer', opacity: submitting ? 0.7 : 1,
                      }}
                    >
                      {submitting ? 'Saving…' : 'Submit resolution'}
                    </button>
                    <button
                      onClick={() => { setResolveOpen(false); setResolveError(null) }}
                      style={{
                        background: 'var(--bg-hover)', border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-sm)', padding: '7px 14px',
                        color: 'var(--text-muted)', fontSize: '12px', cursor: 'pointer',
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div style={{ marginTop: '14px', fontSize: '12px', color: 'var(--accent-success)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>✓</span>
              <span>Resolved — {record.resolution_type}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Main Page ───────────────────────────────────────────────────────────────

const LIMIT = 20

export default function ConflictsPage() {
  const [data, setData] = useState<ConflictsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(1)
  const [filterType, setFilterType] = useState<string>('')
  const [filterResolved, setFilterResolved] = useState<string>('')

  const fetchConflicts = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const params: Record<string, string | number> = { page, limit: LIMIT }
      if (filterType)     params['type'] = filterType
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
  const unresolvedCount = data?.conflicts.filter(c => !c.is_resolved).length ?? 0

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Conflicts</h1>
        <p className="page-subtitle">
          NLI-detected contradictions between retrieved document chunks.
        </p>
      </div>

      {/* ── Summary strip ─────────────────────────────────────────────────── */}
      {data && (
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '20px' }}>
          <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '10px 18px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
            <div style={{ fontSize: '20px', fontWeight: 800, color: 'var(--text-primary)' }}>{data.total}</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Total conflicts</div>
          </div>
          <div style={{ background: 'var(--bg-card)', border: '1px solid rgba(245,158,11,0.3)', borderRadius: 'var(--radius-sm)', padding: '10px 18px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
            <div style={{ fontSize: '20px', fontWeight: 800, color: '#f59e0b' }}>{unresolvedCount}</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Unresolved</div>
          </div>
          <div style={{ background: 'var(--bg-card)', border: '1px solid rgba(52,211,153,0.2)', borderRadius: 'var(--radius-sm)', padding: '10px 18px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
            <div style={{ fontSize: '20px', fontWeight: 800, color: 'var(--accent-success)' }}>{data.total - unresolvedCount}</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Resolved</div>
          </div>
        </div>
      )}

      {/* ── Filter bar ────────────────────────────────────────────────────── */}
      <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '16px', alignItems: 'center' }}>
        <select
          value={filterType}
          onChange={e => { setFilterType(e.target.value); setPage(1) }}
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '7px 12px', color: 'var(--text-primary)', fontSize: '12px', cursor: 'pointer' }}
        >
          <option value="">All types</option>
          <option value="direct_contradiction">Direct Contradiction</option>
          <option value="version_supersession">Version Supersession</option>
          <option value="scope_change">Scope Change</option>
        </select>

        <select
          value={filterResolved}
          onChange={e => { setFilterResolved(e.target.value); setPage(1) }}
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '7px 12px', color: 'var(--text-primary)', fontSize: '12px', cursor: 'pointer' }}
        >
          <option value="">All statuses</option>
          <option value="false">Pending</option>
          <option value="true">Resolved</option>
        </select>

        <button
          onClick={fetchConflicts}
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '7px 14px', color: 'var(--text-secondary)', fontSize: '12px', cursor: 'pointer', fontWeight: 600 }}
        >
          ↺ Refresh
        </button>
      </div>

      {/* ── Content ───────────────────────────────────────────────────────── */}
      {loading && (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '48px' }}>
          <div style={{ width: '32px', height: '32px', border: '3px solid var(--border)', borderTopColor: 'var(--accent-primary)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
        </div>
      )}

      {error && !loading && (
        <div style={{ background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.3)', borderRadius: 'var(--radius-md)', padding: '16px', color: 'var(--accent-error)', fontSize: '13px' }}>
          ⚠ {error}
        </div>
      )}

      {data && !loading && data.conflicts.length === 0 && (
        <div style={{
          background: 'var(--bg-card)', border: '1px dashed var(--border)',
          borderRadius: 'var(--radius-md)', padding: '48px',
          textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px',
        }}>
          <div style={{ fontSize: '40px', marginBottom: '12px' }}>✅</div>
          <div style={{ fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>No conflicts found</div>
          <div>Run queries to detect contradictions between document versions.</div>
        </div>
      )}

      {data && !loading && data.conflicts.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {data.conflicts.map(r => (
            <ConflictRow
              key={r.conflict_id}
              record={r}
              onResolved={() => fetchConflicts()}
            />
          ))}
        </div>
      )}

      {/* ── Pagination ────────────────────────────────────────────────────── */}
      {data && totalPages > 1 && (
        <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', marginTop: '24px' }}>
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page <= 1}
            style={{
              background: 'var(--bg-card)', border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)', padding: '6px 14px',
              color: page <= 1 ? 'var(--text-muted)' : 'var(--text-secondary)',
              fontSize: '12px', cursor: page <= 1 ? 'not-allowed' : 'pointer', fontWeight: 600,
            }}
          >← Prev</button>
          <span style={{ padding: '6px 12px', fontSize: '12px', color: 'var(--text-muted)' }}>
            {page} / {totalPages}
          </span>
          <button
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
            style={{
              background: 'var(--bg-card)', border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)', padding: '6px 14px',
              color: page >= totalPages ? 'var(--text-muted)' : 'var(--text-secondary)',
              fontSize: '12px', cursor: page >= totalPages ? 'not-allowed' : 'pointer', fontWeight: 600,
            }}
          >Next →</button>
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
