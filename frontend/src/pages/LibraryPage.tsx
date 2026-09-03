import { Fragment, useEffect, useRef, useState } from 'react'
import axios from 'axios'

// ── Types ──────────────────────────────────────────────────────────────────

interface DocumentSummary {
  doc_id: string
  title: string
  source_type: string | null
  chunk_count: number
  ingested_at: string
  version_string: string | null
  published_at: string | null
  is_latest: boolean
  parent_doc_id: string | null
}

interface ChunkSummary {
  chunk_id: string
  chunk_index: number
  section_heading: string | null
  content_snippet: string
  token_count: number | null
}

interface DocumentDetail {
  doc_id: string
  title: string
  source_type: string | null
  ingested_at: string
  version_string: string | null
  published_at: string | null
  is_latest: boolean
  parent_doc_id: string | null
  chunks: ChunkSummary[]
}

interface DocumentListResponse {
  items: DocumentSummary[]
  total: number
  page: number
  limit: number
}

interface LineageEntry {
  doc_id: string
  title: string
  version_string: string | null
  published_at: string | null
  ingested_at: string
  is_latest: boolean
}

interface TimelineNode {
  doc_id: string
  title: string
  version_string: string | null
  published_at: string | null
  is_latest: boolean
  parent_doc_id: string | null
  chunk_count: number
  unresolved_conflicts: number
  /** doc_id of the oldest ancestor — nodes sharing it form one version chain. */
  lineage_id: string
  lineage_title: string
  status: 'latest' | 'superseded' | 'conflict'
}

// ── Constants ──────────────────────────────────────────────────────────────

const SOURCE_ICONS: Record<string, string> = { pdf: '📄', md: '📝', txt: '📃' }

const STATUS_COLOR: Record<string, string> = {
  latest:     '#34d399',
  superseded: '#4d5370',
  conflict:   '#f59e0b',
}

const STATUS_LABEL: Record<string, string> = {
  latest:     'Latest',
  superseded: 'Superseded',
  conflict:   'Has conflict',
}

// ── Helpers ────────────────────────────────────────────────────────────────

function fmt(iso: string) {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function fmtDate(iso: string | null) {
  if (!iso) return null
  return new Date(iso).toLocaleDateString(undefined, { dateStyle: 'medium' })
}

function parseDate(s: string | null): number {
  if (!s) return 0
  return new Date(s).getTime()
}

// ── Lineage Strip ──────────────────────────────────────────────────────────

function LineageStrip({ docId, currentDocId }: { docId: string; currentDocId: string }) {
  const [chain, setChain] = useState<LineageEntry[] | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get<LineageEntry[]>(`/api/v1/documents/${docId}/lineage`)
      .then(r => setChain(r.data))
      .catch(() => setChain(null))
      .finally(() => setLoading(false))
  }, [docId])

  if (loading) return <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Loading lineage…</div>
  if (!chain || chain.length <= 1) return null

  return (
    <div style={{ marginBottom: '16px' }}>
      <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.4px', marginBottom: '8px' }}>
        Version Lineage
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexWrap: 'wrap' }}>
        {chain.map((entry, i) => (
          <div key={entry.doc_id} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span
              className="badge"
              style={{
                background: entry.doc_id === currentDocId
                  ? 'var(--accent-subtle)' : entry.is_latest ? 'var(--success-subtle)' : 'var(--bg-elevated)',
                color: entry.doc_id === currentDocId
                  ? 'var(--accent-hover)' : entry.is_latest ? 'var(--success)' : 'var(--text-muted)',
                border: `1px solid ${entry.doc_id === currentDocId ? 'var(--accent-border)' : entry.is_latest ? 'var(--success-border)' : 'var(--border-med)'}`,
                fontFamily: 'ui-monospace, monospace',
              }}
            >
              {entry.version_string ? `v${entry.version_string}` : `#${i + 1}`}
            </span>
            {i < chain.length - 1 && (
              <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>→</span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Document Detail Drawer ─────────────────────────────────────────────────

function DocDetailPanel({ doc, onClose }: { doc: DocumentSummary; onClose: () => void }) {
  const [detail, setDetail] = useState<DocumentDetail | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get<DocumentDetail>(`/api/v1/documents/${doc.doc_id}`)
      .then(r => setDetail(r.data))
      .catch(() => setDetail(null))
      .finally(() => setLoading(false))
  }, [doc.doc_id])

  return (
    <div
      style={{ borderTop: '1px solid var(--border)', padding: '20px 24px', background: 'var(--bg-elevated)', animation: 'fade-in 0.18s ease' }}
      className="animate-fade-in"
    >
      {loading && <div className="loading-center"><div className="spinner" /></div>}

      {detail && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Lineage */}
          <LineageStrip docId={doc.doc_id} currentDocId={doc.doc_id} />

          {/* Metadata row */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', fontSize: '12.5px', color: 'var(--text-muted)' }}>
            {detail.version_string && (
              <span>Version: <strong style={{ color: 'var(--text-primary)' }}>v{detail.version_string}</strong></span>
            )}
            {detail.published_at && (
              <span>Published: <strong style={{ color: 'var(--text-primary)' }}>{fmtDate(detail.published_at)}</strong></span>
            )}
            <span>Chunks: <strong style={{ color: 'var(--accent-hover)' }}>{detail.chunks.length}</strong></span>
            <span>Type: <strong style={{ color: 'var(--text-primary)' }}>{detail.source_type ?? '—'}</strong></span>
            <span>Ingested: <strong style={{ color: 'var(--text-primary)' }}>{fmt(detail.ingested_at)}</strong></span>
          </div>

          {/* Chunk table */}
          <div>
            <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.4px', marginBottom: '8px' }}>
              Chunks ({detail.chunks.length})
            </div>
            <div className="card" style={{ maxHeight: '280px', overflowY: 'auto' }}>
              <table className="table">
                <thead>
                  <tr>
                    <th style={{ width: 40 }}>#</th>
                    <th>Snippet</th>
                    <th style={{ width: 200 }}>Section</th>
                    <th style={{ width: 72, textAlign: 'right' }}>Tokens</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.chunks.map(c => (
                    <tr key={c.chunk_id} style={{ cursor: 'default' }}>
                      <td style={{ color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>{c.chunk_index}</td>
                      <td className="truncate" style={{ maxWidth: 0, color: 'var(--text-secondary)' }}>{c.content_snippet}</td>
                      <td className="truncate" style={{ maxWidth: 0, color: 'var(--text-muted)', fontSize: '12px' }}>
                        {c.section_heading ?? '—'}
                      </td>
                      <td style={{ textAlign: 'right', color: 'var(--accent-hover)', fontWeight: 600 }}>{c.token_count ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div>
            <button className="btn btn-ghost" style={{ fontSize: '12px', padding: '6px 12px' }} onClick={onClose}>
              Collapse ↑
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Documents Tab ──────────────────────────────────────────────────────────

function DocumentsTab() {
  const [docs, setDocs]           = useState<DocumentSummary[]>([])
  const [total, setTotal]         = useState(0)
  const [page, setPage]           = useState(1)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState<string | null>(null)
  const [expandedId, setExpanded] = useState<string | null>(null)

  const LIMIT = 20

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      setError(null)
      try {
        const { data } = await axios.get<DocumentListResponse>('/api/v1/documents', { params: { page, limit: LIMIT } })
        setDocs(data.items)
        setTotal(data.total)
      } catch {
        setError('Failed to load documents. Is the backend running?')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [page])

  const totalPages = Math.ceil(total / LIMIT)

  if (loading) return <div className="loading-center"><div className="spinner spinner-lg" /></div>

  if (error) return (
    <div className="alert alert-danger">
      <svg className="alert-icon" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-11a.75.75 0 00-1.5 0v4a.75.75 0 001.5 0V7zm-.75 7.5a.75.75 0 100-1.5.75.75 0 000 1.5z" clipRule="evenodd"/></svg>
      {error}
    </div>
  )

  if (docs.length === 0) return (
    <div className="card">
      <div className="empty-state">
        <div className="empty-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9 13h6m-3-3v6m-9 1V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2z"/>
          </svg>
        </div>
        <div className="empty-title">No documents yet</div>
        <p className="empty-desc">Upload your first document using the Ingest page to get started.</p>
        <a href="/ingest" className="btn btn-primary" style={{ marginTop: '4px', textDecoration: 'none' }}>
          Go to Ingest →
        </a>
      </div>
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: '12.5px', color: 'var(--text-muted)' }}>
          {total} document{total !== 1 ? 's' : ''}
        </span>
      </div>

      <div className="card">
        <table className="table">
          <thead>
            <tr>
              <th style={{ width: 44 }}>Type</th>
              <th>Title</th>
              <th style={{ width: 80 }}>Status</th>
              <th style={{ width: 72, textAlign: 'right' }}>Chunks</th>
              <th style={{ width: 160 }}>Ingested</th>
              <th style={{ width: 32 }}></th>
            </tr>
          </thead>
          <tbody>
            {docs.map(doc => (
              // Keyed Fragment, not <>: the shorthand cannot take a key, so
              // React warned on every render and could not reconcile rows.
              <Fragment key={doc.doc_id}>
                <tr
                  onClick={() => setExpanded(prev => prev === doc.doc_id ? null : doc.doc_id)}
                  style={{ opacity: doc.is_latest ? 1 : 0.65 }}
                >
                  <td style={{ fontSize: '16px', paddingTop: 10, paddingBottom: 10 }}>
                    {SOURCE_ICONS[doc.source_type ?? ''] ?? '📎'}
                  </td>
                  <td style={{ paddingTop: 10, paddingBottom: 10 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
                      <span style={{ color: 'var(--text-primary)', fontWeight: 500 }} className="truncate">
                        {doc.title}
                      </span>
                      {doc.version_string && (
                        <span className="badge badge-accent" style={{ fontFamily: 'ui-monospace, monospace', fontSize: '10px' }}>
                          v{doc.version_string}
                        </span>
                      )}
                    </div>
                    {doc.published_at && (
                      <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', marginTop: '2px' }}>
                        Published {fmtDate(doc.published_at)}
                      </div>
                    )}
                  </td>
                  <td>
                    {doc.is_latest ? (
                      <span className="badge badge-success"><span className="dot dot-success" />Latest</span>
                    ) : (
                      <span className="badge badge-neutral"><span className="dot dot-muted" />Superseded</span>
                    )}
                  </td>
                  <td style={{ textAlign: 'right', color: 'var(--accent-hover)', fontWeight: 600 }}>
                    {doc.chunk_count}
                  </td>
                  <td style={{ color: 'var(--text-muted)', fontSize: '12px' }}>{fmt(doc.ingested_at)}</td>
                  <td style={{ color: 'var(--text-muted)', textAlign: 'center', fontSize: '11px' }}>
                    {expandedId === doc.doc_id ? '▲' : '▼'}
                  </td>
                </tr>

                {expandedId === doc.doc_id && (
                  <tr>
                    <td colSpan={6} style={{ padding: 0, borderBottom: '1px solid var(--border)' }}>
                      <DocDetailPanel doc={doc} onClose={() => setExpanded(null)} />
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="pagination">
          <button
            className="btn btn-ghost"
            style={{ padding: '6px 14px', fontSize: '12.5px' }}
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
          >
            ← Prev
          </button>
          <span className="pagination-info">{page} / {totalPages}</span>
          <button
            className="btn btn-ghost"
            style={{ padding: '6px 14px', fontSize: '12.5px' }}
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
          >
            Next →
          </button>
        </div>
      )}
    </div>
  )
}

// ── Timeline Tab ───────────────────────────────────────────────────────────

const NODE_R = 13
const LANE_H = 76
// Left padding must fit a right-aligned lane label. At 64px a 16-character
// title overflowed the SVG and was clipped ("…etLib Guide").
const PAD_X  = 150
const PAD_Y  = 44
const LANE_LABEL_MAX = 20

function TimelineTab() {
  const [nodes, setNodes]       = useState<TimelineNode[]>([])
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState<string | null>(null)
  const [selected, setSelected] = useState<TimelineNode | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [width, setWidth]       = useState(700)

  useEffect(() => {
    axios.get<{ nodes: TimelineNode[] }>('/api/v1/analytics/timeline')
      .then(r => setNodes(r.data.nodes))
      .catch(e => setError(axios.isAxiosError(e) ? (e.response?.data?.detail ?? e.message) : 'Unknown error'))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    const obs = new ResizeObserver(entries => {
      for (const e of entries) setWidth(e.contentRect.width || 700)
    })
    if (containerRef.current) obs.observe(containerRef.current)
    return () => obs.disconnect()
  }, [])

  if (loading) return <div className="loading-center"><div className="spinner spinner-lg" /></div>

  if (error) return (
    <div className="alert alert-danger">
      <svg className="alert-icon" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-11a.75.75 0 00-1.5 0v4a.75.75 0 001.5 0V7zm-.75 7.5a.75.75 0 100-1.5.75.75 0 000 1.5z" clipRule="evenodd"/></svg>
      {error}
    </div>
  )

  if (nodes.length === 0) return (
    <div className="card">
      <div className="empty-state">
        <div className="empty-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="9"/>
            <path d="M12 6v6l4 2"/>
          </svg>
        </div>
        <div className="empty-title">No timeline data</div>
        <p className="empty-desc">Ingest documents with version metadata to see the version lineage timeline.</p>
      </div>
    </div>
  )

  // One swimlane per version lineage. This used to group by `domain`, a field
  // no ingest path ever populated, so every document landed in a single lane
  // labelled "unknown". Lineage is real data and is what the timeline is
  // actually for — seeing how one document evolved through its versions.
  const laneMap = new Map<string, { title: string; nodes: TimelineNode[] }>()
  for (const n of nodes) {
    const lane = laneMap.get(n.lineage_id)
    if (lane) lane.nodes.push(n)
    else laneMap.set(n.lineage_id, { title: n.lineage_title, nodes: [n] })
  }
  const lanes = [...laneMap.entries()]

  // Date range → X coordinate
  const allDates = nodes.map(n => parseDate(n.published_at)).filter(Boolean)
  const minDate  = allDates.length ? Math.min(...allDates) : Date.now() - 365 * 86400000
  const maxDate  = allDates.length ? Math.max(...allDates) : Date.now()
  const dateSpan = maxDate - minDate || 1

  const plotW = width - PAD_X * 2
  const svgH  = PAD_Y * 2 + Math.max(1, lanes.length) * LANE_H

  const toX = (dateStr: string | null) => {
    const t = parseDate(dateStr)
    if (!t) return PAD_X + plotW / 2
    return PAD_X + ((t - minDate) / dateSpan) * plotW
  }
  const toY = (i: number) => PAD_Y + i * LANE_H + LANE_H / 2

  const posMap = new Map<string, { x: number; y: number }>()
  lanes.forEach(([, lane], laneIdx) => {
    lane.nodes.forEach(n => posMap.set(n.doc_id, { x: toX(n.published_at), y: toY(laneIdx) }))
  })

  // Evenly spaced ticks labelled by year produced duplicates ("2023, 2023,
  // 2024…") whenever two fell inside the same year. Labels now include the
  // month when the whole range is short, and repeats are dropped.
  const tickCount = 5
  const spansMultipleYears =
    new Date(minDate).getFullYear() !== new Date(maxDate).getFullYear()
  const seenLabels = new Set<string>()
  const ticks = Array.from({ length: tickCount }, (_, i) => {
    const t = minDate + (i / (tickCount - 1)) * dateSpan
    const d = new Date(t)
    const label = spansMultipleYears
      ? d.getFullYear().toString()
      : d.toLocaleDateString(undefined, { month: 'short', year: 'numeric' })
    const duplicate = seenLabels.has(label)
    seenLabels.add(label)
    return { x: PAD_X + (i / (tickCount - 1)) * plotW, label: duplicate ? '' : label }
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Legend */}
      <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
        {Object.entries(STATUS_LABEL).map(([status, label]) => (
          <div key={status} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--text-secondary)' }}>
            <div style={{ width: 10, height: 10, borderRadius: '50%', background: STATUS_COLOR[status] }} />
            {label}
          </div>
        ))}
      </div>

      {/* SVG */}
      <div className="card" ref={containerRef} style={{ overflow: 'auto', padding: '8px' }}>
        <svg width={Math.max(width, 600)} height={svgH} style={{ display: 'block' }}>
          {/* X-axis */}
          <line x1={PAD_X} y1={svgH - PAD_Y} x2={width - PAD_X} y2={svgH - PAD_Y} stroke="var(--border-med)" strokeWidth={1} />

          {/* Ticks */}
          {ticks.map((t, i) => (
            <g key={i}>
              <line x1={t.x} y1={svgH - PAD_Y - 3} x2={t.x} y2={svgH - PAD_Y + 3} stroke="var(--border-med)" strokeWidth={1} />
              <text x={t.x} y={svgH - PAD_Y + 16} textAnchor="middle" fontSize={10} fill="var(--text-muted)">{t.label}</text>
            </g>
          ))}

          {/* One lane per version lineage */}
          {lanes.map(([lineageId, lane], i) => (
            <g key={lineageId}>
              <title>{lane.title}</title>
              <text x={PAD_X - 12} y={toY(i) + 4} textAnchor="end" fontSize={10} fill="var(--text-muted)" fontWeight={600}>
                {lane.title.length > LANE_LABEL_MAX
                  ? lane.title.slice(0, LANE_LABEL_MAX) + '…'
                  : lane.title}
              </text>
              <line x1={PAD_X} y1={toY(i)} x2={width - PAD_X} y2={toY(i)} stroke="var(--border)" strokeWidth={0.5} strokeDasharray="4 4" />
            </g>
          ))}

          {/* Lineage arrows */}
          {nodes.map(n => {
            if (!n.parent_doc_id) return null
            const from = posMap.get(n.parent_doc_id)
            const to   = posMap.get(n.doc_id)
            if (!from || !to) return null
            const dx = to.x - from.x; const dy = to.y - from.y
            const len = Math.sqrt(dx * dx + dy * dy) || 1
            const ux = dx / len; const uy = dy / len
            return (
              <g key={n.doc_id + '_arrow'}>
                <line
                  x1={from.x + ux * NODE_R} y1={from.y + uy * NODE_R}
                  x2={to.x - ux * (NODE_R + 4)} y2={to.y - uy * (NODE_R + 4)}
                  stroke="rgba(99,102,241,0.3)" strokeWidth={1.5}
                />
                <polygon
                  points={`${to.x - ux * NODE_R},${to.y - uy * NODE_R} ${to.x - ux * NODE_R - uy * 4},${to.y - uy * NODE_R + ux * 4} ${to.x - ux * NODE_R + uy * 4},${to.y - uy * NODE_R - ux * 4}`}
                  fill="rgba(99,102,241,0.4)"
                />
              </g>
            )
          })}

          {/* Nodes */}
          {nodes.map(n => {
            const pos = posMap.get(n.doc_id)
            if (!pos) return null
            const color = STATUS_COLOR[n.status] ?? '#6366f1'
            const isSel = selected?.doc_id === n.doc_id
            return (
              <g
                key={n.doc_id}
                onClick={() => setSelected(prev => prev?.doc_id === n.doc_id ? null : n)}
                style={{ cursor: 'pointer' }}
              >
                <circle cx={pos.x} cy={pos.y} r={NODE_R + (isSel ? 4 : 0)} fill={color + '18'} stroke={color} strokeWidth={isSel ? 2.5 : 1.5} />
                <circle cx={pos.x} cy={pos.y} r={NODE_R - 5} fill={color} opacity={0.9} />
                {n.version_string && (
                  <text x={pos.x} y={pos.y - NODE_R - 5} textAnchor="middle" fontSize={9} fill="var(--text-muted)">
                    v{n.version_string}
                  </text>
                )}
              </g>
            )
          })}
        </svg>
      </div>

      {/* Detail panel */}
      {selected && (
        <div
          className="card animate-fade-in"
          style={{ borderColor: `${STATUS_COLOR[selected.status]}44` }}
        >
          <div className="card-body">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '14px' }}>
              <div>
                <div style={{ fontWeight: 700, fontSize: '14px', color: 'var(--text-primary)', marginBottom: '6px' }}>
                  {selected.title}
                </div>
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', alignItems: 'center' }}>
                  {selected.version_string && (
                    <span className="badge badge-accent" style={{ fontFamily: 'ui-monospace, monospace' }}>
                      v{selected.version_string}
                    </span>
                  )}
                  <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{STATUS_LABEL[selected.status]}</span>
                </div>
              </div>
              <button
                className="btn btn-ghost"
                style={{ padding: '4px 8px', fontSize: '13px' }}
                onClick={() => setSelected(null)}
              >
                ✕
              </button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px' }}>
              {[
                { label: 'Lineage',    value: selected.lineage_title },
                { label: 'Published',  value: selected.published_at ?? '—' },
                { label: 'Passages',   value: selected.chunk_count },
                { label: 'Open conflicts', value: selected.unresolved_conflicts },
              ].map(({ label, value }) => (
                <div key={label}>
                  <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.4px', marginBottom: '3px' }}>{label}</div>
                  <div style={{ fontSize: '13px', color: 'var(--text-primary)', fontWeight: 500 }}>{value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Main Page ──────────────────────────────────────────────────────────────

type Tab = 'documents' | 'timeline'

export default function LibraryPage() {
  const [tab, setTab] = useState<Tab>('documents')

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Library</h1>
        <p className="page-subtitle">Browse ingested documents, inspect chunks, and trace version lineage.</p>
      </div>

      <div className="tabs">
        <button
          className={`tab-btn${tab === 'documents' ? ' active' : ''}`}
          onClick={() => setTab('documents')}
        >
          <svg width="14" height="14" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="5" height="14" rx="1"/>
            <rect x="10" y="3" width="7" height="6" rx="1"/>
            <rect x="10" y="11" width="7" height="6" rx="1"/>
          </svg>
          Documents
        </button>
        <button
          className={`tab-btn${tab === 'timeline' ? ' active' : ''}`}
          onClick={() => setTab('timeline')}
        >
          <svg width="14" height="14" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="7"/>
            <path d="M12 8v4l2.5 2.5"/>
            <path d="M5 5L3 3"/>
          </svg>
          Timeline
        </button>
      </div>

      {tab === 'documents' && <DocumentsTab />}
      {tab === 'timeline'  && <TimelineTab />}
    </div>
  )
}
