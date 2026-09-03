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
  const [file,           setFile]           = useState<File | null>(null)
  const [mode,           setMode]           = useState<IngestMode>('new')
  const [title,          setTitle]          = useState('')
  const [parentDocs,     setParentDocs]     = useState<DocumentRoot[]>([])
  const [parentSearch,   setParentSearch]   = useState('')
  const [selectedParent, setSelectedParent] = useState<DocumentRoot | null>(null)
  const [parentsLoading, setParentsLoading] = useState(false)
  const [versionString,  setVersionString]  = useState('')
  const [publishedAt,    setPublishedAt]    = useState('')
  const [versionError,   setVersionError]   = useState('')
  const [loading,        setLoading]        = useState(false)
  const [result,         setResult]         = useState<IngestResult | null>(null)
  const [error,          setError]          = useState<string | null>(null)
  const [dragging,       setDragging]       = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Load parent docs when switching to New Version mode
  useEffect(() => {
    if (mode !== 'version') return
    setParentsLoading(true)
    axios.get<DocumentRoot[]>('/api/v1/documents/roots')
      .then(r => setParentDocs(r.data))
      .catch(() => setParentDocs([]))
      .finally(() => setParentsLoading(false))
  }, [mode])

  // File handling
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

  // Version validation
  const validateVersion = (v: string) => {
    if (!v.trim()) { setVersionError('Version is required.'); return false }
    if (!/^v?\d[\d._\-]*$/.test(v.trim())) {
      setVersionError("Must start with a digit or 'v' followed by digits (e.g. 2.2, v1.13.1)")
      return false
    }
    setVersionError('')
    return true
  }

  // Submit
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
    if (mode === 'version' && selectedParent) form.append('parent_doc_id', selectedParent.doc_id)

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

  const ext = file?.name.split('.').pop()?.toLowerCase()
  const extColor: Record<string, string> = { pdf: '#f87171', md: '#818cf8', txt: '#34d399' }
  const filteredParents = parentDocs.filter(d =>
    d.title.toLowerCase().includes(parentSearch.toLowerCase())
  )
  const canSubmit = !!file &&
    (mode === 'new' ? !!title.trim() : !!selectedParent) &&
    !!versionString.trim() && !versionError &&
    !!publishedAt && !loading

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Ingest Document</h1>
        <p className="page-subtitle">
          Upload a PDF, Markdown, or plain text file. The system parses, chunks, embeds, and indexes it automatically.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '28px', maxWidth: '880px' }}>

        {/* ── Left: Form ── */}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>

          {/* Drop zone */}
          <div
            role="button"
            tabIndex={0}
            aria-label="File upload area"
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={e => e.key === 'Enter' && fileInputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            style={{
              border: `2px dashed ${dragging ? 'var(--accent)' : file ? 'var(--success)' : 'var(--border-med)'}`,
              borderRadius: 'var(--radius-lg)',
              padding: '36px 24px',
              background: dragging ? 'var(--accent-subtle)' : file ? 'var(--success-subtle)' : 'var(--bg-card)',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'border-color 0.2s, background 0.2s',
            }}
          >
            {/* Icon */}
            <div style={{ marginBottom: '12px' }}>
              {file ? (
                <div style={{ fontSize: '32px' }}>{ext && extColor[ext] ? '📄' : '📎'}</div>
              ) : (
                <div style={{
                  width: '48px', height: '48px', borderRadius: 'var(--radius-lg)',
                  background: 'var(--bg-elevated)', border: '1px solid var(--border-med)',
                  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                  color: 'var(--text-muted)',
                }}>
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 15V3M12 3L8 7M12 3L16 7"/>
                    <path d="M4 17v1.5A1.5 1.5 0 005.5 20h13a1.5 1.5 0 001.5-1.5V17"/>
                  </svg>
                </div>
              )}
            </div>

            {file ? (
              <>
                {ext && (
                  <div style={{ marginBottom: '6px' }}>
                    <span className="badge" style={{
                      background: extColor[ext] ? `${extColor[ext]}18` : 'var(--bg-elevated)',
                      color: extColor[ext] ?? 'var(--text-secondary)',
                      border: `1px solid ${extColor[ext] ? `${extColor[ext]}35` : 'var(--border-med)'}`,
                      textTransform: 'uppercase', letterSpacing: '0.5px',
                    }}>
                      .{ext}
                    </span>
                  </div>
                )}
                <p style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '4px' }}>
                  {file.name}
                </p>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  {(file.size / 1024).toFixed(1)} KB — click to change
                </p>
              </>
            ) : (
              <>
                <p style={{ fontSize: '13.5px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                  Drop a file here, or click to browse
                </p>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Accepted formats: .pdf · .md · .txt
                </p>
              </>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.md,.txt"
              style={{ display: 'none' }}
              onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
          </div>

          {/* Mode toggle */}
          <div className="field">
            <span className="field-label">Document type</span>
            <div style={{ display: 'flex', gap: '8px' }}>
              {(['new', 'version'] as IngestMode[]).map(m => (
                <label
                  key={m}
                  style={{
                    flex: 1,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '10px 14px',
                    borderRadius: 'var(--radius-md)',
                    border: `1.5px solid ${mode === m ? 'var(--accent)' : 'var(--border-med)'}`,
                    background: mode === m ? 'var(--accent-subtle)' : 'var(--bg-elevated)',
                    cursor: 'pointer',
                    transition: 'border-color 0.15s, background 0.15s',
                    fontSize: '13px',
                    fontWeight: mode === m ? 600 : 400,
                    color: mode === m ? 'var(--accent-hover)' : 'var(--text-secondary)',
                  }}
                >
                  <input
                    type="radio"
                    name="ingest-mode"
                    value={m}
                    checked={mode === m}
                    onChange={() => { setMode(m); setSelectedParent(null); setParentSearch('') }}
                    style={{ accentColor: 'var(--accent)', width: '14px', height: '14px' }}
                  />
                  {m === 'new' ? 'New document' : 'New version'}
                </label>
              ))}
            </div>
          </div>

          {/* New Version: parent selector */}
          {mode === 'version' && (
            <div className="field animate-fade-in">
              <label className="field-label">Parent document *</label>
              {parentsLoading ? (
                <div style={{ fontSize: '13px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div className="spinner" /> Loading documents…
                </div>
              ) : parentDocs.length === 0 ? (
                <div className="alert alert-warning" style={{ fontSize: '12.5px' }}>
                  No existing documents found. Ingest a document first.
                </div>
              ) : (
                <>
                  <input
                    id="parent-search"
                    className="input"
                    type="text"
                    placeholder="Search documents…"
                    value={parentSearch}
                    onChange={e => setParentSearch(e.target.value)}
                  />
                  <div style={{
                    maxHeight: '180px', overflowY: 'auto',
                    border: '1px solid var(--border-med)',
                    borderRadius: 'var(--radius-md)',
                    background: 'var(--bg-elevated)',
                  }}>
                    {filteredParents.length === 0 ? (
                      <div style={{ padding: '10px 14px', fontSize: '12.5px', color: 'var(--text-muted)' }}>
                        No documents match.
                      </div>
                    ) : filteredParents.map(doc => (
                      <div
                        key={doc.doc_id}
                        onClick={() => { setSelectedParent(doc); setParentSearch('') }}
                        style={{
                          padding: '9px 14px',
                          cursor: 'pointer',
                          borderBottom: '1px solid var(--border)',
                          background: selectedParent?.doc_id === doc.doc_id ? 'var(--accent-subtle)' : 'transparent',
                          transition: 'background 0.12s',
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
                          <span className="badge badge-accent" style={{ fontFamily: 'ui-monospace, monospace', fontSize: '10px' }}>
                            v{doc.version_string}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                  {selectedParent && (
                    <div className="alert alert-success" style={{ fontSize: '12.5px' }}>
                      ✓ Title inherited: <strong>{selectedParent.title}</strong>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* New document title */}
          {mode === 'new' && (
            <div className="field animate-fade-in">
              <label htmlFor="ingest-title" className="field-label">Document title *</label>
              <input
                id="ingest-title"
                className="input"
                type="text"
                value={title}
                onChange={e => setTitle(e.target.value)}
                placeholder="e.g. PyTorch Docs"
                required
              />
            </div>
          )}

          {/* Version */}
          <div className="field">
            <label htmlFor="ingest-version" className="field-label">Version *</label>
            <input
              id="ingest-version"
              className={`input${versionError ? ' input-error' : ''}`}
              type="text"
              value={versionString}
              onChange={e => { setVersionString(e.target.value); if (versionError) validateVersion(e.target.value) }}
              onBlur={e => validateVersion(e.target.value)}
              placeholder="e.g. 2.2 or v1.13.1 or 2024-03-01"
              required
            />
            {versionError && <span className="field-error">{versionError}</span>}
          </div>

          {/* Published date */}
          <div className="field">
            <label htmlFor="ingest-published" className="field-label">Published date *</label>
            <input
              id="ingest-published"
              className="input"
              type="date"
              value={publishedAt}
              onChange={e => setPublishedAt(e.target.value)}
              required
              style={{ colorScheme: 'dark' }}
            />
          </div>

          {/* Submit */}
          <button
            id="ingest-submit-btn"
            type="submit"
            className="btn btn-primary"
            disabled={!canSubmit}
            style={{ padding: '11px 20px', fontSize: '14px' }}
          >
            {loading ? (
              <>
                <div className="spinner" style={{ width: '15px', height: '15px', borderColor: 'rgba(255,255,255,0.3)', borderTopColor: '#fff' }} />
                Processing…
              </>
            ) : (
              <>
                <svg width="15" height="15" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M10 13V4M10 4L6.5 7.5M10 4L13.5 7.5"/>
                  <path d="M3 14.5v1A1.5 1.5 0 004.5 17h11A1.5 1.5 0 0017 15.5v-1"/>
                </svg>
                Ingest Document
              </>
            )}
          </button>
        </form>

        {/* ── Right: Result / Info panel ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', paddingTop: '2px' }}>

          {/* Success result */}
          {result && (
            <div className="card animate-fade-in">
              <div className="card-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ color: 'var(--success)', fontSize: '16px' }}>✓</span>
                  <span className="card-title" style={{ color: 'var(--success)' }}>Ingested successfully</span>
                </div>
                <span className="badge badge-success">v{result.version_string}</span>
              </div>
              <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  {[
                    { label: 'Chunks created', value: result.chunks_created.toString(), highlight: true },
                    { label: 'Status',          value: result.is_latest ? 'Latest' : 'Older version', highlight: false },
                    { label: 'Published',        value: result.published_at,             highlight: false },
                    { label: 'Ingested at',      value: new Date(result.ingested_at).toLocaleTimeString(), highlight: false },
                  ].map(({ label, value, highlight }) => (
                    <div
                      key={label}
                      style={{
                        background: highlight ? 'var(--success-subtle)' : 'var(--bg-elevated)',
                        border: `1px solid ${highlight ? 'var(--success-border)' : 'var(--border)'}`,
                        borderRadius: 'var(--radius-md)',
                        padding: '10px 14px',
                      }}
                    >
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.4px', marginBottom: '4px', fontWeight: 600 }}>
                        {label}
                      </div>
                      <div style={{
                        fontSize: highlight ? '26px' : '13px',
                        fontWeight: highlight ? 700 : 500,
                        color: highlight ? 'var(--success)' : 'var(--text-primary)',
                        letterSpacing: highlight ? '-0.5px' : 0,
                      }}>
                        {value}
                      </div>
                    </div>
                  ))}
                </div>

                {result.lineage_message && result.lineage_message !== 'New document lineage started.' && (
                  <div className="alert alert-warning" style={{ fontSize: '12.5px' }}>
                    🔗 {result.lineage_message}
                  </div>
                )}

                {result.lineage_message === 'New document lineage started.' && (
                  <div className="alert alert-info" style={{ fontSize: '12.5px' }}>
                    🌱 New lineage started — future versions will be linked here.
                  </div>
                )}

                <div style={{ display: 'flex', gap: '10px', paddingTop: '2px' }}>
                  <a href="/library" className="btn btn-secondary" style={{ fontSize: '13px', textDecoration: 'none' }}>
                    View in Library →
                  </a>
                  <a href="/query" className="btn btn-ghost" style={{ fontSize: '13px', textDecoration: 'none' }}>
                    Go to Query
                  </a>
                </div>
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="alert alert-danger animate-fade-in">
              <svg className="alert-icon" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-11a.75.75 0 00-1.5 0v4a.75.75 0 001.5 0V7zm-.75 7.5a.75.75 0 100-1.5.75.75 0 000 1.5z" clipRule="evenodd"/>
              </svg>
              <div>
                <div style={{ fontWeight: 600, marginBottom: '3px' }}>Ingestion failed</div>
                <div style={{ fontSize: '12.5px', opacity: 0.9, wordBreak: 'break-word' }}>{error}</div>
              </div>
            </div>
          )}

          {/* Info panel */}
          {!result && !error && (
            <div className="card">
              <div className="card-header">
                <span className="card-title">What happens on ingest?</span>
              </div>
              <div className="card-body">
                <ol style={{
                  paddingLeft: '18px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px',
                  fontSize: '13px',
                  color: 'var(--text-secondary)',
                  lineHeight: 1.65,
                }}>
                  {[
                    'Version and date are validated from your input',
                    'File is parsed → raw text + headings extracted',
                    'Text split into ≤400-token chunks',
                    'Each chunk embedded with all-MiniLM-L6-v2',
                    'Vectors indexed in FAISS + BM25',
                    'Version lineage linked to parent (if selected)',
                    'Everything stored in PostgreSQL',
                  ].map((step, i) => (
                    <li key={i}>{step}</li>
                  ))}
                </ol>
              </div>
            </div>
          )}
        </div>
      </div>

    </div>
  )
}
