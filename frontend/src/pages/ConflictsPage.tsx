import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

// ── Types ───────────────────────────────────────────────────────────────────

interface ChunkInfo {
  chunk_id: string
  doc_id: string
  snippet: string
  section_heading: string | null
  doc_title: string
  version_string: string | null
  published_at: string | null
  is_latest: boolean
}

type ResolutionType = 'prefer_source' | 'both_valid'

interface ConflictRecord {
  conflict_id: string
  conflict_type: string
  nli_score: number
  detected_at: string
  detected_during: string | null
  needs_review: boolean
  is_resolved: boolean
  resolution_type: ResolutionType | null
  preferred_chunk_id: string | null
  resolution_note: string | null
  resolved_at: string | null
  /** For a version change: chunk_a is the earlier version, chunk_b the later. */
  chunk_a: ChunkInfo
  chunk_b: ChunkInfo
}

interface ConflictsResponse {
  total: number
  page: number
  limit: number
  conflicts: ConflictRecord[]
  total_needs_review: number
  total_resolved: number
  total_version_changes: number
}

type Status = '' | 'needs_review' | 'resolved' | 'version_change'

// ── Helpers ─────────────────────────────────────────────────────────────────

// The detector emits exactly these two types, decided by version lineage alone
// (services/conflict_detector _classify). Only a contradiction is reviewed.
const VERSION_CHANGE = 'version_supersession'

const TYPE_INFO: Record<string, { label: string; help: string; badge: string }> = {
  version_supersession: {
    label: 'Changed between versions',
    help: 'Two versions of the same document say different things. Each is correct for its own validity period, so the version history already explains it and no review is needed.',
    badge: 'badge-accent',
  },
  direct_contradiction: {
    label: 'Contradiction',
    help: 'Two unrelated documents disagree. Until a reviewer decides, current answers prefer the source published more than 90 days later, and otherwise show both.',
    badge: 'badge-danger',
  },
}

function typeInfo(t: string) {
  return TYPE_INFO[t] ?? { label: t, help: '', badge: 'badge-accent' }
}

function nliClass(score: number) {
  if (score >= 0.85) return 'badge-danger'
  if (score >= 0.70) return 'badge-warning'
  return 'badge-accent'
}

function fmtDate(iso: string) {
  try { return new Date(iso).toLocaleString() } catch { return iso }
}

/** "Handbook v2.0 · 2023-03-15" — enough to tell two sources apart. */
function sourceName(c: ChunkInfo): string {
  return [c.doc_title + (c.version_string ? ` v${c.version_string}` : ''), c.published_at]
    .filter(Boolean)
    .join(' · ')
}

function sideLabels(record: ConflictRecord): [string, string] {
  return record.conflict_type === VERSION_CHANGE
    ? ['Earlier version', 'Later version']
    : ['Source A', 'Source B']
}

/** One-line summary of the two sides. Never infers "same document" from a title. */
function describePair(record: ConflictRecord): string {
  const { chunk_a: a, chunk_b: b } = record
  const v = (c: ChunkInfo) => (c.version_string ? `v${c.version_string}` : 'unversioned')
  if (record.conflict_type === VERSION_CHANGE) return `${a.doc_title}: ${v(a)} → ${v(b)}`
  return `${sourceName(a)}  vs.  ${sourceName(b)}`
}

/** The decision exactly as stored — nothing implied that was not recorded. */
function describeDecision(record: ConflictRecord): string {
  if (record.resolution_type === 'both_valid') return 'Both are correct — not a real contradiction'
  const winner = [record.chunk_a, record.chunk_b].find(c => c.chunk_id === record.preferred_chunk_id)
  return winner ? `Prefer ${sourceName(winner)}` : 'Unrecognised decision'
}

// ── Chunk Side ───────────────────────────────────────────────────────────────

function ChunkSide({ info, label, verdict }: {
  info: ChunkInfo
  label: string
  verdict?: 'preferred' | 'set_aside'
}) {
  return (
    <div
      className="card"
      style={{
        flex: 1, minWidth: 0,
        borderColor: verdict === 'preferred' ? 'var(--success-border)'
                   : verdict === 'set_aside' ? 'var(--danger-border)' : undefined,
      }}
    >
      <div style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', alignItems: 'center' }}>
          <span style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            {label}
          </span>
          {verdict === 'preferred' && <span className="badge badge-success">Preferred by reviewer</span>}
          {verdict === 'set_aside' && (
            <span className="badge badge-danger" title="Excluded from answers about the present. Questions about the past still see it.">
              Set aside for current answers
            </span>
          )}
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
              <span className="dot dot-success" />Latest version
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

// ── Decision form ────────────────────────────────────────────────────────────

/** The choice in the form: a chunk id to prefer, or 'both_valid'. */
type Choice = string | 'both_valid'

function DecisionForm({ record, onSaved, onCancel }: {
  record: ConflictRecord
  onSaved: (updated: ConflictRecord) => void
  onCancel: () => void
}) {
  const [a, b] = [record.chunk_a, record.chunk_b]
  const [labelA, labelB] = sideLabels(record)
  // Changing a decision starts from the recorded one; a new review starts
  // with nothing selected, so a winner is never chosen by default.
  const initial: Choice | null =
    record.resolution_type === 'both_valid' ? 'both_valid' : record.preferred_chunk_id
  const [choice,     setChoice]     = useState<Choice | null>(initial)
  const [note,       setNote]       = useState(record.resolution_note ?? '')
  const [submitting, setSubmitting] = useState(false)
  const [error,      setError]      = useState<string | null>(null)

  const options: { value: Choice; label: string; hint: string }[] = [
    { value: a.chunk_id, label: `Prefer ${labelA}: ${sourceName(a)}`, hint: `${labelB} is wrong or out of date.` },
    { value: b.chunk_id, label: `Prefer ${labelB}: ${sourceName(b)}`, hint: `${labelA} is wrong or out of date.` },
    { value: 'both_valid', label: 'Both are correct', hint: 'They apply to different scopes or situations, or do not actually contradict.' },
  ]

  const submit = async () => {
    if (choice === null) return
    setSubmitting(true)
    setError(null)
    const body = choice === 'both_valid'
      ? { resolution_type: 'both_valid', resolution_note: note }
      : { resolution_type: 'prefer_source', preferred_chunk_id: choice, resolution_note: note }
    try {
      const { data } = await axios.post<ConflictRecord>(`/api/v1/conflicts/${record.conflict_id}/resolve`, body)
      onSaved(data)
    } catch (err: unknown) {
      setError(axios.isAxiosError(err) ? (err.response?.data?.detail ?? err.message) : 'Unexpected error')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="card" style={{ background: 'var(--bg-elevated)' }}>
      <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>Which is right?</div>

        <div role="radiogroup" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {options.map(o => (
            <label key={o.value} style={{ display: 'flex', gap: '10px', alignItems: 'flex-start', cursor: 'pointer', fontSize: '12.5px' }}>
              <input
                type="radio"
                name={`decision-${record.conflict_id}`}
                checked={choice === o.value}
                onChange={() => setChoice(o.value)}
                style={{ marginTop: '3px' }}
              />
              <span>
                <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{o.label}</span>
                <span style={{ display: 'block', color: 'var(--text-muted)', fontSize: '11.5px' }}>{o.hint}</span>
              </span>
            </label>
          ))}
        </div>

        <p style={{ fontSize: '11.5px', color: 'var(--text-muted)', margin: 0 }}>
          Preferring a source removes the other one from answers about the present.
          Questions about the past still see both, with your decision explained.
          Not sure? Leave it pending.
        </p>

        <div className="field">
          <label className="field-label">Note (optional, shown to the answering model)</label>
          <textarea
            className="input"
            value={note}
            onChange={e => setNote(e.target.value)}
            placeholder="What did you check?"
            rows={2}
          />
        </div>

        {error && <div className="alert alert-danger" style={{ fontSize: '12px' }}>{error}</div>}

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={submit}
            disabled={submitting || choice === null}
            className="btn btn-primary"
            style={{ fontSize: '12.5px', padding: '7px 16px' }}
          >
            {submitting ? (
              <>
                <div className="spinner" style={{ width: '13px', height: '13px', borderColor: 'rgba(255,255,255,0.3)', borderTopColor: '#fff' }} />
                Saving…
              </>
            ) : 'Save decision'}
          </button>
          <button onClick={onCancel} className="btn btn-ghost" style={{ fontSize: '12.5px' }}>
            Cancel
          </button>
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
  const [record,   setRecord]   = useState<ConflictRecord>(initialRecord)
  const [expanded, setExpanded] = useState(false)
  const [deciding, setDeciding] = useState(false)

  const isVersionChange = record.conflict_type === VERSION_CHANGE
  const info = typeInfo(record.conflict_type)
  const labels = sideLabels(record)

  const verdictOf = (c: ChunkInfo): 'preferred' | 'set_aside' | undefined => {
    if (!record.is_resolved || record.resolution_type !== 'prefer_source') return undefined
    return c.chunk_id === record.preferred_chunk_id ? 'preferred' : 'set_aside'
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
        <div style={{ minWidth: 0 }}>
          <div style={{ fontWeight: 600, fontSize: '13px', color: 'var(--text-primary)' }} className="truncate">
            {describePair(record)}
          </div>
          {record.is_resolved && (
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }} className="truncate">
              {describeDecision(record)}
            </div>
          )}
        </div>

        <span className={`badge ${info.badge}`} title={info.help}>{info.label}</span>

        {/* One decimal place: scores routinely sit at 0.997, and rounding
            those to a flat "100%" claims a certainty the model never states. */}
        <span
          className={`badge ${nliClass(record.nli_score)}`}
          style={{ fontFamily: 'ui-monospace, monospace' }}
          title="How confident the contradiction model is that these two passages disagree."
        >
          {(record.nli_score * 100).toFixed(1)}% confident
        </span>

        {isVersionChange ? (
          <span className="badge badge-accent" title={info.help}>No review needed</span>
        ) : record.is_resolved ? (
          <span className="badge badge-success">Resolved</span>
        ) : (
          <span className="badge badge-warning">Needs review</span>
        )}

        <span style={{
          color: 'var(--text-muted)', fontSize: '11px',
          transform: expanded ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.2s',
        }}>▾</span>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div style={{ borderTop: '1px solid var(--border)', padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: '14px' }}>

          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', fontSize: '12px', color: 'var(--text-muted)' }}>
            <span>Detected: <strong style={{ color: 'var(--text-secondary)' }}>{fmtDate(record.detected_at)}</strong></span>
            {record.detected_during && (
              <span>During query: <code style={{ fontSize: '11px', color: 'var(--text-secondary)', background: 'var(--bg-elevated)', padding: '1px 5px', borderRadius: 'var(--radius-sm)' }}>
                {record.detected_during.slice(0, 8)}…
              </code></span>
            )}
          </div>

          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            {[record.chunk_a, record.chunk_b].map((c, i) => (
              <ChunkSide key={c.chunk_id} info={c} label={labels[i]} verdict={verdictOf(c)} />
            ))}
          </div>

          {isVersionChange ? (
            <p style={{ fontSize: '12.5px', color: 'var(--text-muted)', margin: 0 }}>
              {info.help} Questions about a given time are answered from the version valid then.
            </p>
          ) : deciding ? (
            <DecisionForm
              record={record}
              onCancel={() => setDeciding(false)}
              onSaved={updated => { setRecord(updated); setDeciding(false); onResolved?.(updated) }}
            />
          ) : record.is_resolved ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12.5px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--success)', fontWeight: 600 }}>
                <span className="dot dot-success" />
                Decision: {describeDecision(record)}
              </div>
              {record.resolution_note && (
                <div style={{ color: 'var(--text-secondary)' }}>Note: {record.resolution_note}</div>
              )}
              {record.resolved_at && (
                <div style={{ color: 'var(--text-muted)', fontSize: '11.5px' }}>Decided {fmtDate(record.resolved_at)}</div>
              )}
              <div>
                <button onClick={() => setDeciding(true)} className="btn btn-ghost" style={{ fontSize: '12.5px', padding: '6px 12px' }}>
                  Change decision
                </button>
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {record.resolution_note && (
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Earlier reviewer note: {record.resolution_note}</div>
              )}
              <div>
                <button onClick={() => setDeciding(true)} className="btn btn-secondary" style={{ fontSize: '12.5px', padding: '7px 16px' }}>
                  Review
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Main Page ────────────────────────────────────────────────────────────────

const LIMIT = 20

const STATUS_OPTIONS: { value: Status; label: string }[] = [
  { value: '',               label: 'All' },
  { value: 'needs_review',   label: 'Needs review' },
  { value: 'resolved',       label: 'Resolved' },
  { value: 'version_change', label: 'Version changes' },
]

export default function ConflictsPage() {
  const [data,    setData]    = useState<ConflictsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)
  const [page,    setPage]    = useState(1)
  const [status,  setStatus]  = useState<Status>('')

  const fetchConflicts = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const params: Record<string, string | number> = { page, limit: LIMIT }
      if (status) params['status'] = status
      const { data: res } = await axios.get<ConflictsResponse>('/api/v1/conflicts', { params })
      setData(res)
    } catch (err: unknown) {
      setError(axios.isAxiosError(err) ? (err.response?.data?.detail ?? err.message) : 'Unexpected error')
    } finally {
      setLoading(false)
    }
  }, [page, status])

  useEffect(() => { fetchConflicts() }, [fetchConflicts])

  const totalPages = data ? Math.ceil(data.total / LIMIT) : 1

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Conflicts</h1>
        <p className="page-subtitle">
          Passages that contradict each other, found while answering questions.
          Contradictions between unrelated documents need a decision; changes
          between versions of one document are explained by its history.
        </p>
      </div>

      {/* Corpus-wide counts, from the server rather than the current page. */}
      {data && (
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '24px' }}>
          {[
            { label: 'Needs review',    value: data.total_needs_review },
            { label: 'Resolved',        value: data.total_resolved },
            { label: 'Version changes', value: data.total_version_changes },
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
          value={status}
          onChange={e => { setStatus(e.target.value as Status); setPage(1) }}
          style={{ width: 'auto', padding: '7px 32px 7px 12px' }}
        >
          {STATUS_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>

        <button onClick={fetchConflicts} className="btn btn-secondary" style={{ fontSize: '12.5px', padding: '7px 14px' }}>
          ↺ Refresh
        </button>
      </div>

      {loading && (
        <div className="loading-center"><div className="spinner spinner-lg" /></div>
      )}

      {error && !loading && (
        <div className="alert alert-danger">
          <svg className="alert-icon" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-11a.75.75 0 00-1.5 0v4a.75.75 0 001.5 0V7zm-.75 7.5a.75.75 0 100-1.5.75.75 0 000 1.5z" clipRule="evenodd"/></svg>
          {error}
        </div>
      )}

      {data && !loading && data.conflicts.length === 0 && (
        <div className="card">
          <div className="empty-state">
            <div className="empty-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>
            </div>
            <div className="empty-title">
              {status ? 'Nothing here' : 'No conflicts detected'}
            </div>
            <p className="empty-desc">
              {status === 'needs_review'
                ? 'Every contradiction has a decision.'
                : status
                  ? 'Try another filter.'
                  : 'Run queries to detect contradictions between documents.'}
            </p>
          </div>
        </div>
      )}

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
