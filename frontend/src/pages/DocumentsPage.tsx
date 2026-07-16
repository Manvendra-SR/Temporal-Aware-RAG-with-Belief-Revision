import { useEffect, useState } from 'react'
import axios from 'axios'

// ── Types ──────────────────────────────────────────────────────────────────

interface DocumentSummary {
  doc_id: string
  title: string
  domain: string
  source_type: string | null
  chunk_count: number
  ingested_at: string
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
  domain: string
  source_type: string | null
  ingested_at: string
  chunks: ChunkSummary[]
}

interface DocumentListResponse {
  items: DocumentSummary[]
  total: number
  page: number
  limit: number
}

// ── Constants ──────────────────────────────────────────────────────────────

const DOMAIN_LABELS: Record<string, string> = {
  general: 'General', pytorch_docs: 'PyTorch', python_docs: 'Python',
  npm_docs: 'npm', arxiv_cs: 'arXiv CS', legal: 'Legal',
}

const SOURCE_ICONS: Record<string, string> = { pdf: '📄', md: '📝', txt: '📃' }

function fmt(iso: string) {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

// ── Component ──────────────────────────────────────────────────────────────

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentSummary[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [domainFilter, setDomainFilter] = useState('')
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
        if (domainFilter) params.domain = domainFilter
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
  }, [page, domainFilter])

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
        <p className="page-subtitle">Browse ingested documents and inspect their chunks.</p>
      </div>

      {/* ── Toolbar ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
        <select
          id="docs-domain-filter"
          value={domainFilter}
          onChange={e => { setDomainFilter(e.target.value); setPage(1) }}
          style={{
            background: 'var(--bg-card)', border: '1px solid var(--border-light)',
            borderRadius: 'var(--radius-sm)', padding: '7px 12px',
            color: 'var(--text-primary)', fontSize: '13px', cursor: 'pointer',
          }}
        >
          <option value="">All Domains</option>
          {Object.entries(DOMAIN_LABELS).map(([v, l]) => (
            <option key={v} value={v} style={{ background: 'var(--bg-secondary)' }}>{l}</option>
          ))}
        </select>

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
                {['Type', 'Title', 'Domain', 'Chunks', 'Ingested At', ''].map(h => (
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
                    }}
                    onMouseEnter={e => { if (expandedId !== doc.doc_id) (e.currentTarget as HTMLElement).style.background = 'var(--bg-tertiary)' }}
                    onMouseLeave={e => { if (expandedId !== doc.doc_id) (e.currentTarget as HTMLElement).style.background = 'transparent' }}
                  >
                    <td style={{ padding: '12px 14px', color: 'var(--text-muted)', fontSize: '18px' }}>
                      {SOURCE_ICONS[doc.source_type ?? ''] ?? '📎'}
                    </td>
                    <td style={{ padding: '12px 14px', color: 'var(--text-primary)', fontWeight: 500, maxWidth: '280px' }}>
                      <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{doc.title}</div>
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <span style={{
                        background: 'rgba(99,102,241,0.12)', color: 'var(--accent-primary-hover)',
                        borderRadius: '999px', padding: '2px 9px', fontSize: '11px', fontWeight: 600,
                      }}>
                        {DOMAIN_LABELS[doc.domain] ?? doc.domain}
                      </span>
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

                  {/* ── Expanded chunk list ── */}
                  {expandedId === doc.doc_id && (
                    <tr key={`${doc.doc_id}-detail`} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td colSpan={6} style={{ padding: '0 14px 16px' }}>
                        {detailLoading && (
                          <div style={{ color: 'var(--text-muted)', padding: '16px 0', fontSize: '13px' }}>Loading chunks…</div>
                        )}
                        {detail && (
                          <div style={{ marginTop: '12px' }}>
                            <div style={{
                              display: 'grid',
                              gridTemplateColumns: '48px 1fr 200px 64px',
                              gap: '0',
                              background: 'var(--bg-tertiary)',
                              borderRadius: 'var(--radius-sm)',
                              overflow: 'hidden',
                              border: '1px solid var(--border)',
                            }}>
                              {/* Chunk table header */}
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
