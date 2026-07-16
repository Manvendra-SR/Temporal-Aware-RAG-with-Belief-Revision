import { useRef, useState } from 'react'
import axios from 'axios'

const DOMAINS = [
  { value: 'general',      label: 'General' },
  { value: 'pytorch_docs', label: 'PyTorch Docs' },
  { value: 'python_docs',  label: 'Python Docs' },
  { value: 'npm_docs',     label: 'npm Docs' },
  { value: 'arxiv_cs',     label: 'arXiv CS' },
  { value: 'legal',        label: 'Legal' },
]

interface IngestResult {
  doc_id: string
  title: string
  domain: string
  chunks_created: number
  ingested_at: string
}

export default function IngestPage() {
  const [file, setFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [domain, setDomain] = useState('general')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<IngestResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File) => {
    setFile(f)
    setResult(null)
    setError(null)
    // Auto-fill title from filename if empty
    if (!title) {
      setTitle(f.name.replace(/\.[^.]+$/, '').replace(/[-_]/g, ' '))
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file || !title.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    const form = new FormData()
    form.append('file', file)
    form.append('title', title.trim())
    form.append('domain', domain)

    try {
      const { data } = await axios.post<IngestResult>('/api/v1/ingest', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120_000,   // embedding can take a few seconds
      })
      setResult(data)
      setFile(null)
      setTitle('')
      if (fileInputRef.current) fileInputRef.current.value = ''
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail
        setError(detail ?? `Error ${err.response?.status}: ${err.message}`)
      } else {
        setError('Unexpected error. Check the backend logs.')
      }
    } finally {
      setLoading(false)
    }
  }

  const fileLabel = file
    ? `${file.name} (${(file.size / 1024).toFixed(1)} KB)`
    : 'Drop a file here or click to browse'

  const ext = file?.name.split('.').pop()?.toLowerCase()
  const extColor: Record<string, string> = { pdf: '#f87171', md: '#818cf8', txt: '#34d399' }

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Ingest Documents</h1>
        <p className="page-subtitle">
          Upload a PDF, Markdown, or TXT file to add it to the knowledge base.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', maxWidth: '860px' }}>
        {/* ── Upload form ── */}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

          {/* Drop zone */}
          <div
            role="button"
            tabIndex={0}
            aria-label="File upload drop zone"
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={e => e.key === 'Enter' && fileInputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            style={{
              border: `2px dashed ${dragging ? 'var(--accent-primary)' : file ? 'var(--accent-success)' : 'var(--border-light)'}`,
              borderRadius: 'var(--radius-md)',
              padding: '32px 20px',
              background: dragging ? 'rgba(99,102,241,0.06)' : 'var(--bg-card)',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'border-color 0.2s, background 0.2s',
            }}
          >
            <div style={{ fontSize: '36px', marginBottom: '10px' }}>
              {file ? (ext && extColor[ext] ? '📄' : '📎') : '⬆'}
            </div>
            {file && ext && (
              <span style={{
                display: 'inline-block',
                padding: '2px 10px',
                borderRadius: '999px',
                background: extColor[ext] ? `${extColor[ext]}22` : 'var(--bg-hover)',
                color: extColor[ext] ?? 'var(--text-secondary)',
                fontWeight: 700,
                fontSize: '11px',
                marginBottom: '6px',
                textTransform: 'uppercase',
              }}>
                .{ext}
              </span>
            )}
            <p style={{ color: file ? 'var(--text-primary)' : 'var(--text-secondary)', fontSize: '13px', margin: 0 }}>
              {fileLabel}
            </p>
            <p style={{ color: 'var(--text-muted)', fontSize: '11px', marginTop: '6px' }}>
              Accepted: .pdf · .md · .txt
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.md,.txt"
              style={{ display: 'none' }}
              onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
          </div>

          {/* Title */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label htmlFor="ingest-title" style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Title *
            </label>
            <input
              id="ingest-title"
              type="text"
              value={title}
              onChange={e => setTitle(e.target.value)}
              placeholder="PyTorch v2.2 Docs"
              required
              style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border-light)',
                borderRadius: 'var(--radius-sm)',
                padding: '9px 12px',
                color: 'var(--text-primary)',
                fontSize: '14px',
                outline: 'none',
                transition: 'border-color 0.2s',
              }}
              onFocus={e => (e.target.style.borderColor = 'var(--accent-primary)')}
              onBlur={e => (e.target.style.borderColor = 'var(--border-light)')}
            />
          </div>

          {/* Domain */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label htmlFor="ingest-domain" style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Domain
            </label>
            <select
              id="ingest-domain"
              value={domain}
              onChange={e => setDomain(e.target.value)}
              style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border-light)',
                borderRadius: 'var(--radius-sm)',
                padding: '9px 12px',
                color: 'var(--text-primary)',
                fontSize: '14px',
                outline: 'none',
                cursor: 'pointer',
              }}
            >
              {DOMAINS.map(d => (
                <option key={d.value} value={d.value} style={{ background: 'var(--bg-secondary)' }}>
                  {d.label}
                </option>
              ))}
            </select>
          </div>

          {/* Submit */}
          <button
            type="submit"
            id="ingest-submit-btn"
            disabled={!file || !title.trim() || loading}
            style={{
              background: loading || !file || !title.trim()
                ? 'var(--bg-hover)'
                : 'linear-gradient(135deg, var(--accent-primary), #4f46e5)',
              border: 'none',
              borderRadius: 'var(--radius-sm)',
              padding: '11px 20px',
              color: loading || !file || !title.trim() ? 'var(--text-muted)' : 'white',
              fontSize: '14px',
              fontWeight: 600,
              cursor: !file || !title.trim() || loading ? 'not-allowed' : 'pointer',
              transition: 'opacity 0.2s',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
            }}
          >
            {loading ? (
              <>
                <span style={{ display: 'inline-block', width: '14px', height: '14px', border: '2px solid rgba(255,255,255,0.3)', borderTopColor: 'white', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />
                Processing…
              </>
            ) : (
              <>⬆ Ingest Document</>
            )}
          </button>
        </form>

        {/* ── Result / Error panel ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {result && (
            <div style={{
              background: 'var(--bg-card)',
              border: '1px solid rgba(52,211,153,0.4)',
              borderRadius: 'var(--radius-md)',
              padding: '20px',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '20px' }}>✅</span>
                <span style={{ fontWeight: 700, color: 'var(--accent-success)', fontSize: '15px' }}>
                  Ingested successfully
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                {[
                  { label: 'Chunks Created', value: result.chunks_created.toString(), highlight: true },
                  { label: 'Domain', value: result.domain, highlight: false },
                  { label: 'Doc ID', value: result.doc_id.slice(0, 12) + '…', highlight: false },
                  { label: 'Ingested At', value: new Date(result.ingested_at).toLocaleTimeString(), highlight: false },
                ].map(({ label, value, highlight }) => (
                  <div key={label} style={{
                    background: highlight ? 'rgba(52,211,153,0.08)' : 'var(--bg-tertiary)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '10px 14px',
                    border: highlight ? '1px solid rgba(52,211,153,0.2)' : '1px solid var(--border)',
                  }}>
                    <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>{label}</div>
                    <div style={{ fontSize: highlight ? '24px' : '13px', fontWeight: highlight ? 700 : 500, color: highlight ? 'var(--accent-success)' : 'var(--text-primary)' }}>{value}</div>
                  </div>
                ))}
              </div>

              <a
                href="/documents"
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: '6px',
                  color: 'var(--accent-primary-hover)', fontSize: '13px', fontWeight: 500,
                  textDecoration: 'none',
                }}
              >
                → View in Documents
              </a>
            </div>
          )}

          {error && (
            <div style={{
              background: 'rgba(248,113,113,0.08)',
              border: '1px solid rgba(248,113,113,0.3)',
              borderRadius: 'var(--radius-md)',
              padding: '16px 20px',
              display: 'flex',
              gap: '10px',
              alignItems: 'flex-start',
            }}>
              <span style={{ fontSize: '18px', flexShrink: 0, marginTop: '1px' }}>⚠</span>
              <div>
                <div style={{ fontWeight: 600, color: 'var(--accent-error)', marginBottom: '4px' }}>Ingestion failed</div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '13px', wordBreak: 'break-word' }}>{error}</div>
              </div>
            </div>
          )}

          {/* Hint card when idle */}
          {!result && !error && (
            <div style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)',
              padding: '20px',
            }}>
              <div style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '8px' }}>What happens on ingest?</strong>
                <ol style={{ paddingLeft: '18px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <li>File is parsed → raw text + headings</li>
                  <li>Text split into ≤400-token chunks</li>
                  <li>Each chunk embedded (all-MiniLM-L6-v2)</li>
                  <li>Vectors indexed in FAISS</li>
                  <li>Text indexed in BM25</li>
                  <li>Stored in PostgreSQL</li>
                </ol>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Spin animation */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
