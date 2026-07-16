export default function HomePage() {
  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">
          <span className="gradient-text">Temporal RAG</span> with Belief Revision
        </h1>
        <p className="page-subtitle">
          A research system that understands document versions, detects conflicts, and reasons about time.
        </p>
      </div>

      {/* Phase roadmap cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '16px' }}>
        {[
          { phase: 1, title: 'Project Skeleton',              status: 'done',    icon: '✅', desc: 'FastAPI + React + PostgreSQL skeleton' },
          { phase: 2, title: 'Ingestion Pipeline',            status: 'pending', icon: '🔜', desc: 'PDF/MD/TXT → chunks → FAISS + BM25' },
          { phase: 3, title: 'Hybrid Retrieval + Generation', status: 'pending', icon: '🔜', desc: 'BM25 + dense + RRF merge + LLM answer' },
          { phase: 4, title: 'Temporal Metadata',             status: 'pending', icon: '🔜', desc: 'Version lineage, published_at, half-lives' },
          { phase: 5, title: 'Temporal Reranking',            status: 'pending', icon: '🔜', desc: 'Exponential decay scoring, version pinning' },
          { phase: 6, title: 'Conflict Detection',            status: 'pending', icon: '🔜', desc: 'NLI-based contradiction detection' },
          { phase: 7, title: 'Belief Revision',               status: 'pending', icon: '🔜', desc: 'Decision tree, confidence scoring, context annotation' },
        ].map(({ phase, title, status, icon, desc }) => (
          <div
            key={phase}
            style={{
              background: 'var(--bg-card)',
              border: `1px solid ${status === 'done' ? 'rgba(52, 211, 153, 0.3)' : 'var(--border)'}`,
              borderRadius: 'var(--radius-md)',
              padding: '20px',
              display: 'flex',
              flexDirection: 'column',
              gap: '8px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '18px' }}>{icon}</span>
              <span style={{
                fontSize: '10px',
                fontWeight: 700,
                padding: '2px 8px',
                borderRadius: '999px',
                background: status === 'done' ? 'rgba(52,211,153,0.15)' : 'rgba(99,102,241,0.12)',
                color: status === 'done' ? 'var(--accent-success)' : 'var(--accent-primary-hover)',
                border: `1px solid ${status === 'done' ? 'rgba(52,211,153,0.3)' : 'rgba(99,102,241,0.3)'}`,
              }}>
                Phase {phase}
              </span>
            </div>
            <h3 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>{title}</h3>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>{desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
