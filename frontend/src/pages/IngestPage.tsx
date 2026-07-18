import { useEffect, useRef, useState } from 'react'
import axios from 'axios'

// ── Types ──────────────────────────────────────────────────────────────────

interface IngestResult {
  doc_id: string
  title: string
  version_string: string
  published_at: string
  is_latest: boolean
  chunks_created: number
  ingested_at: string
  lineage_message: string
}

interface DocumentRoot {
  doc_id: string
  title: string
  version_string: string | null
}

type IngestMode = 'new' | 'version'

// ── Component ──────────────────────────────────────────────────────────────

export default function IngestPage() {
  const [file, setFile] = useState<File | null>(null)
  const [mode, setMode] = useState<IngestMode>('new')

  // New Document fields
  const [title, setTitle] = useState('')

  // New Version fields
  const [parentDocs, setParentDocs] = useState<DocumentRoot[]>([])
  const [parentSearch, setParentSearch] = useState('')
  const [selectedParent, setSelectedParent] = useState<DocumentRoot | null>(null)
  const [parentsLoading, setParentsLoading] = useState(false)

  // Shared required fields
  const [versionString, setVersionString] = useState('')
  const [publishedAt, setPublishedAt] = useState('')

  // Validation errors
  const [versionError, setVersionError] = useState('')

  // Submission state
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<IngestResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // ── Load parent docs when switching to New Version mode ──────────────────
  useEffect(() => {
    if (mode !== 'version') return
    setParentsLoading(true)
    axios.get<DocumentRoot[]>('/api/v1/documents/roots')
      .then(r => setParentDocs(r.data))
      .catch(() => setParentDocs([]))
      .finally(() => setParentsLoading(false))
  }, [mode])

  // ── File handling ─────────────────────────────────────────────────────────
  const handleFile = (f: File) => {
    setFile(f)
    setResult(null)
    setError(null)
    if (mode === 'new' && !title) {
      setTitle(f.name.replace(/\.[^.]+$/, '').replace(/[-_]/g, ' '))
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  // ── Version validation ────────────────────────────────────────────────────
  const validateVersion = (v: string) => {
    if (!v.trim()) { setVersionError('Version is required.'); return false }
    if (!/^v?\d[\d._\-]*$/.test(v.trim())) {
      setVersionError("Must start with a digit or 'v' followed by digits (e.g. 2.2, v1.13.1)")
      return false
    }
    setVersionError('')
    return true
  }

  // ── Submit ────────────────────────────────────────────────────────────────
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) return
    if (mode === 'new' && !title.trim()) return
    if (mode === 'version' && !selectedParent) return
    if (!validateVersion(versionString)) return
    if (!publishedAt) return

    setLoading(true)
    setError(null)
    setResult(null)

    const resolvedTitle = mode === 'version' ? selectedParent!.title : title.trim()

    const form = new FormData()
    form.append('file', file)
    form.append('title', resolvedTitle)
    form.append('version_string', versionString.trim())
    form.append('published_at', publishedAt)
    if (mode === 'version' && selectedParent) {
      form.append('parent_doc_id', selectedParent.doc_id)
    }

    try {
      const { data } = await axios.post<IngestResult>('/api/v1/ingest', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120_000,
      })
      setResult(data)
      setFile(null)
      setTitle('')
      setVersionString('')
      setPublishedAt('')
      setSelectedParent(null)
      setParentSearch('')
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

  // ── Derived state ─────────────────────────────────────────────────────────
  const fileLabel = file
    ? `${file.name} (${(file.size / 1024).toFixed(1)} KB)`
    : 'Drop a file here or click to browse'

  const ext = file?.name.split('.').pop()?.toLowerCase()
  const extColor: Record<string, string> = { pdf: '#f87171', md: '#818cf8', txt: '#34d399' }

  const filteredParents = parentDocs.filter(d =>
    d.title.toLowerCase().includes(parentSearch.toLowerCase())
  )

  const canSubmit = !!file &&
    (mode === 'new' ? !!title.trim() : !!selectedParent) &&
    !!versionString.trim() && !versionError &&
    !!publishedAt &&
    !loading

  // ── Styles ────────────────────────────────────────────────────────────────
  const inputStyle = {
    background: 'var(--bg-card)',
    border: '1px solid var(--border-light)',
    borderRadius: 'var(--radius-sm)',
    padding: '9px 12px',
    color: 'var(--text-primary)',
    fontSize: '14px',
    outline: 'none',
    transition: 'border-color 0.2s',
    width: '100%',
    boxSizing: 'border-box' as const,
  }
  const labelStyle = {
    fontSize: '12px',
    fontWeight: 600 as const,
    color: 'var(--text-secondary)' as const,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
  }
  const fieldStyle = { display: 'flex', flexDirection: 'column' as const, gap: '6px' }

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

          {/* ── Mode selector ── */}
          <div style={fieldStyle}>
            <span style={labelStyle}>Document Type *</span>
            <div style={{ display: 'flex', gap: '10px' }}>
              {(['new', 'version'] as IngestMode[]).map(m => (
                <label
                  key={m}
                  style={{
                    flex: 1,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '10px 14px',
                    borderRadius: 'var(--radius-sm)',
                    border: `1.5px solid ${mode === m ? 'var(--accent-primary)' : 'var(--border-light)'}`,
                    background: mode === m ? 'rgba(99,102,241,0.08)' : 'var(--bg-card)',
                    cursor: 'pointer',
                    transition: 'border-color 0.2s, background 0.2s',
                    fontSize: '13px',
                    fontWeight: mode === m ? 600 : 400,
                    color: mode === m ? 'var(--accent-primary-hover)' : 'var(--text-secondary)',
                  }}
                >
                  <input
                    type="radio"
                    name="ingest-mode"
                    value={m}
                    checked={mode === m}
                    onChange={() => { setMode(m); setSelectedParent(null); setParentSearch('') }}
                    style={{ accentColor: 'var(--accent-primary)' }}
                  />
                  {m === 'new' ? '🌱 New Document' : '🔗 New Version'}
                </label>
              ))}
            </div>
          </div>

          {/* ── New Version: parent selector ── */}
          {mode === 'version' && (
            <div style={{ ...fieldStyle, animation: 'fadeIn 0.15s ease' }}>
              <label style={labelStyle}>Parent Document *</label>
              {parentsLoading ? (
                <div style={{ fontSize: '13px', color: 'var(--text-muted)', padding: '8px 0' }}>
                  Loading existing documents…
                </div>
              ) : parentDocs.length === 0 ? (
                <div style={{
                  padding: '12px', borderRadius: 'var(--radius-sm)',
                  background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.25)',
                  fontSize: '12px', color: '#fbbf24',
                }}>
                  No existing documents found. Ingest a document first before uploading a new version.
                </div>
              ) : (
                <>
                  <input
                    id="parent-search"
                    type="text"
                    placeholder="Search documents…"
                    value={parentSearch}
                    onChange={e => setParentSearch(e.target.value)}
                    style={{ ...inputStyle, marginBottom: '6px' }}
                    onFocus={e => (e.target.style.borderColor = 'var(--accent-primary)')}
                    onBlur={e => (e.target.style.borderColor = 'var(--border-light)')}
                  />
                  <div style={{
                    maxHeight: '180px',
                    overflowY: 'auto',
                    border: '1px solid var(--border-light)',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-card)',
                  }}>
                    {filteredParents.length === 0 ? (
                      <div style={{ padding: '10px 14px', fontSize: '12px', color: 'var(--text-muted)' }}>
                        No documents match your search.
                      </div>
                    ) : filteredParents.map(doc => (
                      <div
                        key={doc.doc_id}
                        onClick={() => { setSelectedParent(doc); setParentSearch('') }}
                        style={{
                          padding: '9px 14px',
                          cursor: 'pointer',
                          borderBottom: '1px solid var(--border)',
                          background: selectedParent?.doc_id === doc.doc_id
                            ? 'rgba(99,102,241,0.12)'
                            : 'transparent',
                          transition: 'background 0.15s',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                        }}
                        onMouseEnter={e => { if (selectedParent?.doc_id !== doc.doc_id) (e.currentTarget as HTMLElement).style.background = 'var(--bg-hover)' }}
                        onMouseLeave={e => { if (selectedParent?.doc_id !== doc.doc_id) (e.currentTarget as HTMLElement).style.background = 'transparent' }}
                      >
                        <span style={{ fontSize: '13px', color: 'var(--text-primary)', fontWeight: selectedParent?.doc_id === doc.doc_id ? 600 : 400 }}>
                          {doc.title}
                        </span>
                        {doc.version_string && (
                          <span style={{
                            padding: '1px 8px', borderRadius: '999px',
                            background: 'rgba(129,140,248,0.15)', color: '#818cf8',
                            fontSize: '11px', fontWeight: 700, fontFamily: 'monospace',
                          }}>
                            v{doc.version_string}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                  {selectedParent && (
                    <div style={{
                      padding: '8px 12px', borderRadius: 'var(--radius-sm)',
                      background: 'rgba(52,211,153,0.08)', border: '1px solid rgba(52,211,153,0.2)',
                      fontSize: '12px', color: '#34d399', display: 'flex', alignItems: 'center', gap: '6px',
                    }}>
                      <span>✓</span>
                      <span>
                        Title will be inherited: <strong>{selectedParent.title}</strong>
                      </span>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* ── New Document: title field ── */}
          {mode === 'new' && (
            <div style={{ ...fieldStyle, animation: 'fadeIn 0.15s ease' }}>
              <label htmlFor="ingest-title" style={labelStyle}>Title *</label>
              <input
                id="ingest-title"
                type="text"
                value={title}
                onChange={e => setTitle(e.target.value)}
                placeholder="PyTorch Docs"
                required
                style={inputStyle}
                onFocus={e => (e.target.style.borderColor = 'var(--accent-primary)')}
                onBlur={e => (e.target.style.borderColor = 'var(--border-light)')}
              />
            </div>
          )}

          {/* ── Version string ── */}
          <div style={fieldStyle}>
            <label htmlFor="ingest-version" style={labelStyle}>Version *</label>
            <input
              id="ingest-version"
              type="text"
              value={versionString}
              onChange={e => { setVersionString(e.target.value); if (versionError) validateVersion(e.target.value) }}
              onBlur={e => validateVersion(e.target.value)}
              placeholder="e.g. 2.2 or 1.13.1 or 2024-03-01"
              required
              style={{
                ...inputStyle,
                borderColor: versionError ? 'var(--accent-error)' : 'var(--border-light)',
              }}
              onFocus={e => (e.target.style.borderColor = versionError ? 'var(--accent-error)' : 'var(--accent-primary)')}
            />
            {versionError && (
              <span style={{ fontSize: '11px', color: 'var(--accent-error)' }}>{versionError}</span>
            )}
          </div>

          {/* ── Published date ── */}
          <div style={fieldStyle}>
            <label htmlFor="ingest-published" style={labelStyle}>Published Date *</label>
            <input
              id="ingest-published"
              type="date"
              value={publishedAt}
              onChange={e => setPublishedAt(e.target.value)}
              required
              style={{ ...inputStyle, colorScheme: 'dark' }}
              onFocus={e => (e.target.style.borderColor = 'var(--accent-primary)')}
              onBlur={e => (e.target.style.borderColor = 'var(--border-light)')}
            />
          </div>

          {/* ── Submit ── */}
          <button
            type="submit"
            id="ingest-submit-btn"
            disabled={!canSubmit}
            style={{
              background: canSubmit
                ? 'linear-gradient(135deg, var(--accent-primary), #4f46e5)'
                : 'var(--bg-hover)',
              border: 'none',
              borderRadius: 'var(--radius-sm)',
              padding: '11px 20px',
              color: canSubmit ? 'white' : 'var(--text-muted)',
              fontSize: '14px',
              fontWeight: 600,
              cursor: canSubmit ? 'pointer' : 'not-allowed',
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
                  { label: 'Version', value: result.version_string, highlight: false },
                  { label: 'Published', value: result.published_at, highlight: false },
                  { label: 'Status', value: result.is_latest ? '✅ Latest' : '📦 Older version', highlight: false },
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

              {result.lineage_message && result.lineage_message !== 'New document lineage started.' && (
                <div style={{
                  background: 'rgba(251,191,36,0.08)',
                  border: '1px solid rgba(251,191,36,0.3)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '10px 14px',
                  fontSize: '12px',
                  color: '#fbbf24',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '8px',
                }}>
                  <span style={{ flexShrink: 0 }}>🔗</span>
                  <span>{result.lineage_message}</span>
                </div>
              )}

              {result.lineage_message === 'New document lineage started.' && (
                <div style={{
                  background: 'rgba(99,102,241,0.08)',
                  border: '1px solid rgba(99,102,241,0.2)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '10px 14px',
                  fontSize: '12px',
                  color: 'var(--accent-primary-hover)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                }}>
                  <span>🌱</span>
                  <span>New lineage started — future versions will be linked here.</span>
                </div>
              )}

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
                  <li>Version and date are validated from your input</li>
                  <li>File is parsed → raw text + headings</li>
                  <li>Text split into ≤400-token chunks</li>
                  <li>Each chunk embedded (all-MiniLM-L6-v2)</li>
                  <li>Vectors indexed in FAISS + BM25</li>
                  <li>Version lineage linked to parent document (if selected)</li>
                  <li>Stored in PostgreSQL</li>
                </ol>
              </div>
            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  )
}
