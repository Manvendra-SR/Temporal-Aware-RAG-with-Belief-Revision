import { useEffect, useState } from 'react'
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

// ── Constants ──────────────────────────────────────────────────────────────

const SOURCE_ICONS: Record<string, string> = { pdf: '📄', md: '📝', txt: '📃' }

function fmt(iso: string) {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function fmtDate(iso: string | null) {
  if (!iso) return null
  return new Date(iso).toLocaleDateString(undefined, { dateStyle: 'medium' })
}

// ── Sub-components ─────────────────────────────────────────────────────────

function VersionBadge({ version }: { version: string | null }) {
  if (!version) return null
  return (
    <span style={{
      display: 'inline-block',
      padding: '1px 7px',
      borderRadius: '999px',
      background: 'rgba(129,140,248,0.15)',
      color: '#818cf8',
      fontSize: '11px',
      fontWeight: 700,
      letterSpacing: '0.3px',
      marginLeft: '6px',
      fontFamily: 'monospace',
    }}>
      v{version}
    </span>
  )
}

function StatusBadge({ isLatest }: { isLatest: boolean }) {
  return isLatest ? (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '4px',
      padding: '1px 8px',
      borderRadius: '999px',
      background: 'rgba(52,211,153,0.12)',
      color: '#34d399',
      fontSize: '10px',
      fontWeight: 700,
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
    }}>
      <span style={{ width: '5px', height: '5px', borderRadius: '50%', background: '#34d399', display: 'inline-block' }} />
      Latest
    </span>
  ) : (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '4px',
      padding: '1px 8px',
      borderRadius: '999px',
      background: 'rgba(156,163,175,0.12)',
      color: '#9ca3af',
      fontSize: '10px',
      fontWeight: 700,
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
    }}>
      <span style={{ width: '5px', height: '5px', borderRadius: '50%', background: '#9ca3af', display: 'inline-block' }} />
      Superseded
    </span>
  )
}

function LineageStrip({ docId, currentDocId }: { docId: string; currentDocId: string }) {
  const [chain, setChain] = useState<LineageEntry[] | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get<LineageEntry[]>(`/api/v1/documents/${docId}/lineage`)
      .then(r => setChain(r.data))
      .catch(() => setChain(null))
      .finally(() => setLoading(false))
  }, [docId])

  if (loading) return <div style={{ color: 'var(--text-muted)', fontSize: '12px', padding: '8px 0' }}>Loading lineage…</div>
  if (!chain || chain.length <= 1) return null

  return (
    <div style={{ marginBottom: '16px' }}>
      <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '8px' }}>
        Version Lineage
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexWrap: 'wrap' }}>
        {chain.map((entry, i) => (
          <div key={entry.doc_id} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <div style={{
              padding: '3px 10px',
              borderRadius: '999px',
              fontSize: '11px',
              fontWeight: 700,
              fontFamily: 'monospace',
              border: entry.doc_id === currentDocId
                ? '1.5px solid var(--accent-primary)'
                : '1px solid var(--border-light)',
              background: entry.doc_id === currentDocId
                ? 'rgba(99,102,241,0.12)'
                : entry.is_latest
                  ? 'rgba(52,211,153,0.08)'
                  : 'var(--bg-tertiary)',
              color: entry.doc_id === currentDocId
                ? 'var(--accent-primary-hover)'
                : entry.is_latest
                  ? '#34d399'
                  : 'var(--text-muted)',
            }}>
              {entry.version_string ? `v${entry.version_string}` : `#${i + 1}`}
            </div>
            {i < chain.length - 1 && (
              <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>→</span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Component ──────────────────────────────────────────────────────────────

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentSummary[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Expanded row detail
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [detail, setDetail] = useState<DocumentDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  const LIMIT = 20

  // Fetch document list
  useEffect(() => {
    const fetch = async () => {
      setLoading(true)
      setError(null)
      try {
        const params: Record<string, string | number> = { page, limit: LIMIT }
        const { data } = await axios.get<DocumentListResponse>('/api/v1/documents', { params })
        setDocs(data.items)
        setTotal(data.total)
      } catch {
        setError('Failed to load documents. Is the backend running?')
      } finally {
        setLoading(false)
      }
    }
    fetch()
  }, [page])

  // Toggle row expansion
  const toggleRow = async (doc_id: string) => {
    if (expandedId === doc_id) {
      setExpandedId(null)
      setDetail(null)
      return
    }
    setExpandedId(doc_id)
    setDetail(null)
    setDetailLoading(true)
    try {
      const { data } = await axios.get<DocumentDetail>(`/api/v1/documents/${doc_id}`)
      setDetail(data)
    } catch {
      setDetail(null)
    } finally {
      setDetailLoading(false)
    }
  }

  const totalPages = Math.ceil(total / LIMIT)

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Documents</h1>
        <p className="page-subtitle">Browse ingested documents, inspect chunks, and trace version lineage.</p>
      </div>

      {/* ── Toolbar ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
        <span style={{ color: 'var(--text-muted)', fontSize: '13px', marginLeft: 'auto' }}>
          {total} document{total !== 1 ? 's' : ''}
        </span>
      </div>

      {/* ── Error ── */}
      {error && (
        <div style={{
          background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.3)',
          borderRadius: 'var(--radius-md)', padding: '14px 18px',
          color: 'var(--accent-error)', fontSize: '13px', marginBottom: '16px',
        }}>
          ⚠ {error}
        </div>
      )}

      {/* ── Empty state ── */}
      {!loading && docs.length === 0 && !error && (
        <div className="coming-soon" style={{ minHeight: '40vh' }}>
          <div className="coming-soon-icon">📭</div>
          <h2>No documents yet</h2>
          <p>Upload your first document on the Ingest page.</p>
          <a href="/ingest" style={{ color: 'var(--accent-primary-hover)', textDecoration: 'none', fontWeight: 600 }}>
            → Go to Ingest
          </a>
        </div>
      )}

      {/* ── Table ── */}
      {docs.length > 0 && (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-tertiary)' }}>
                {['Type', 'Title & Version', 'Chunks', 'Ingested At', ''].map(h => (
                  <th key={h} style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 600, fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {docs.map(doc => (
                <>
                  <tr
                    key={doc.doc_id}
                    onClick={() => toggleRow(doc.doc_id)}
                    style={{
                      borderBottom: expandedId === doc.doc_id ? 'none' : '1px solid var(--border)',
                      cursor: 'pointer',
                      background: expandedId === doc.doc_id ? 'var(--bg-hover)' : 'transparent',
                      transition: 'background 0.15s',
                      opacity: doc.is_latest ? 1 : 0.72,
                    }}
                    onMouseEnter={e => { if (expandedId !== doc.doc_id) (e.currentTarget as HTMLElement).style.background = 'var(--bg-tertiary)' }}
                    onMouseLeave={e => { if (expandedId !== doc.doc_id) (e.currentTarget as HTMLElement).style.background = 'transparent' }}
                  >
                    <td style={{ padding: '12px 14px', color: 'var(--text-muted)', fontSize: '18px' }}>
                      {SOURCE_ICONS[doc.source_type ?? ''] ?? '📎'}
                    </td>
                    <td style={{ padding: '12px 14px', maxWidth: '300px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '4px' }}>
                        <span style={{ color: 'var(--text-primary)', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '200px' }}>
                          {doc.title}
                        </span>
                        <VersionBadge version={doc.version_string} />
                      </div>
                      <div style={{ marginTop: '4px' }}>
                        <StatusBadge isLatest={doc.is_latest} />
                      </div>
                    </td>
                    <td style={{ padding: '12px 14px', color: 'var(--accent-secondary)', fontWeight: 600 }}>
                      {doc.chunk_count}
                    </td>
                    <td style={{ padding: '12px 14px', color: 'var(--text-muted)' }}>
                      {fmt(doc.ingested_at)}
                    </td>
                    <td style={{ padding: '12px 14px', color: 'var(--text-muted)', textAlign: 'right' }}>
                      {expandedId === doc.doc_id ? '▲' : '▼'}
                    </td>
                  </tr>

                  {/* ── Expanded detail ── */}
                  {expandedId === doc.doc_id && (
                    <tr key={`${doc.doc_id}-detail`} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td colSpan={5} style={{ padding: '0 14px 16px' }}>
                        {detailLoading && (
                          <div style={{ color: 'var(--text-muted)', padding: '16px 0', fontSize: '13px' }}>Loading…</div>
                        )}
                        {detail && (
                          <div style={{ marginTop: '12px' }}>

                            {/* Lineage strip */}
                            <LineageStrip docId={doc.doc_id} currentDocId={doc.doc_id} />

                            {/* Doc metadata bar */}
                            <div style={{
                              display: 'flex', flexWrap: 'wrap', gap: '16px',
                              padding: '10px 14px',
                              background: 'var(--bg-tertiary)',
                              borderRadius: 'var(--radius-sm)',
                              marginBottom: '12px',
                              fontSize: '12px',
                              color: 'var(--text-muted)',
                              border: '1px solid var(--border)',
                            }}>
                              {detail.version_string && (
                                <span>Version: <strong style={{ color: 'var(--text-primary)' }}>{detail.version_string}</strong></span>
                              )}
                              {detail.published_at && (
                                <span>Published: <strong style={{ color: 'var(--text-primary)' }}>{fmtDate(detail.published_at)}</strong></span>
                              )}
                              <span>Chunks: <strong style={{ color: 'var(--accent-secondary)' }}>{detail.chunks.length}</strong></span>
                              <span>Type: <strong style={{ color: 'var(--text-primary)' }}>{detail.source_type ?? '—'}</strong></span>
                            </div>

                            {/* Chunk table */}
                            <div style={{
                              display: 'grid',
                              gridTemplateColumns: '48px 1fr 200px 64px',
                              background: 'var(--bg-tertiary)',
                              borderRadius: 'var(--radius-sm)',
                              overflow: 'hidden',
                              border: '1px solid var(--border)',
                            }}>
                              {['#', 'Snippet', 'Heading', 'Tokens'].map(h => (
                                <div key={h} style={{ padding: '7px 12px', fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', borderBottom: '1px solid var(--border)' }}>
                                  {h}
                                </div>
                              ))}
                              {detail.chunks.map((c, i) => (
                                <>
                                  <div key={`${c.chunk_id}-idx`} style={{ padding: '8px 12px', color: 'var(--text-muted)', fontSize: '12px', borderBottom: i < detail.chunks.length - 1 ? '1px solid var(--border)' : 'none', background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)' }}>
                                    {c.chunk_index}
                                  </div>
                                  <div key={`${c.chunk_id}-snip`} style={{ padding: '8px 12px', color: 'var(--text-secondary)', fontSize: '12px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', borderBottom: i < detail.chunks.length - 1 ? '1px solid var(--border)' : 'none', background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)' }}>
                                    {c.content_snippet}
                                  </div>
                                  <div key={`${c.chunk_id}-head`} style={{ padding: '8px 12px', color: 'var(--text-muted)', fontSize: '11px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', borderBottom: i < detail.chunks.length - 1 ? '1px solid var(--border)' : 'none', background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)' }}>
                                    {c.section_heading ?? '—'}
                                  </div>
                                  <div key={`${c.chunk_id}-tok`} style={{ padding: '8px 12px', color: 'var(--accent-secondary)', fontSize: '12px', fontWeight: 600, borderBottom: i < detail.chunks.length - 1 ? '1px solid var(--border)' : 'none', background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)' }}>
                                    {c.token_count ?? '—'}
                                  </div>
                                </>
                              ))}
                            </div>
                          </div>
                        )}
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Pagination ── */}
      {totalPages > 1 && (
        <div style={{ display: 'flex', gap: '8px', justifyContent: 'center', marginTop: '20px' }}>
          <button
            id="docs-prev-btn"
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '6px 16px', color: page === 1 ? 'var(--text-muted)' : 'var(--text-primary)', cursor: page === 1 ? 'not-allowed' : 'pointer' }}
          >
            ← Prev
          </button>
          <span style={{ padding: '6px 12px', color: 'var(--text-secondary)', fontSize: '13px' }}>
            {page} / {totalPages}
          </span>
          <button
            id="docs-next-btn"
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '6px 16px', color: page === totalPages ? 'var(--text-muted)' : 'var(--text-primary)', cursor: page === totalPages ? 'not-allowed' : 'pointer' }}
          >
            Next →
          </button>
        </div>
      )}
    </div>
  )
}
