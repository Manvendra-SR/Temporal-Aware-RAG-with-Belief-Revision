import { useCallback, useEffect, useState } from 'react'
import { NavLink, Outlet, useOutletContext } from 'react-router-dom'
import axios from 'axios'

// ── Types ──────────────────────────────────────────────────────────────────

type HealthStatus = 'checking' | 'connected' | 'disconnected'

export interface Stats {
  total_docs: number
  total_chunks: number
  total_queries: number
  total_conflicts: number
  unresolved_conflicts: number
  total_lineages: number
  superseded_docs: number
}

export interface ShellContext {
  stats: Stats | null
  /** Re-fetch corpus stats — call after ingesting or resolving something. */
  refreshStats: () => void
}

/**
 * Corpus stats, shared from the shell so pages can react to an empty corpus
 * without each issuing its own request.
 */
export function useShell(): ShellContext {
  return useOutletContext<ShellContext>()
}

// ── SVG Icons ──────────────────────────────────────────────────────────────

const UploadIcon = () => (
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" width="17" height="17">
    <path d="M10 13V4M10 4L6.5 7.5M10 4L13.5 7.5"/>
    <path d="M3 14.5v1A1.5 1.5 0 004.5 17h11A1.5 1.5 0 0017 15.5v-1"/>
  </svg>
)

const LibraryIcon = () => (
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" width="17" height="17">
    <rect x="3" y="3" width="5" height="14" rx="1"/>
    <rect x="10" y="3" width="7" height="6" rx="1"/>
    <rect x="10" y="11" width="7" height="6" rx="1"/>
  </svg>
)

const QueryIcon = () => (
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" width="17" height="17">
    <circle cx="9" cy="9" r="5.5"/>
    <path d="M14.5 14.5L17 17"/>
    <path d="M9 6.5v.5M9 11v.5" strokeWidth="1.5"/>
  </svg>
)

const ConflictIcon = () => (
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" width="17" height="17">
    <path d="M10 3L17.3 16H2.7L10 3z"/>
    <path d="M10 9v3M10 13.5v.5" strokeWidth="1.5"/>
  </svg>
)

const BrainIcon = () => (
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" width="16" height="16">
    <path d="M6.5 4.5C5.5 4.5 4 5.5 4 7.5c0 1.2.6 2 1.5 2.5-.9.5-1.5 1.3-1.5 2.5 0 2 1.5 3 2.5 3h7c1 0 2.5-1 2.5-3 0-1.2-.6-2-1.5-2.5.9-.5 1.5-1.3 1.5-2.5 0-2-1.5-3-2.5-3"/>
    <path d="M10 4.5v11M7 7.5h6M7 12.5h6"/>
  </svg>
)

// ── Nav Items ──────────────────────────────────────────────────────────────

const NAV = [
  { to: '/ingest',    label: 'Ingest',    icon: <UploadIcon /> },
  { to: '/library',   label: 'Library',   icon: <LibraryIcon /> },
  { to: '/query',     label: 'Query',     icon: <QueryIcon /> },
  { to: '/conflicts', label: 'Conflicts', icon: <ConflictIcon /> },
]

// ── Component ──────────────────────────────────────────────────────────────

export default function Layout() {
  const [health, setHealth] = useState<HealthStatus>('checking')
  const [stats, setStats] = useState<Stats | null>(null)

  const loadStats = useCallback(async () => {
    try {
      const { data } = await axios.get<Stats>('/api/v1/analytics/overview', { timeout: 6000 })
      setStats(data)
    } catch {
      // Non-critical: the sidebar simply omits the counts.
    }
  }, [])

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const { data } = await axios.get('/health', { timeout: 4000 })
        setHealth(data.db_connected ? 'connected' : 'disconnected')
      } catch {
        setHealth('disconnected')
      }
    }

    checkHealth()
    loadStats()

    const healthInterval = setInterval(checkHealth, 10_000)
    const statsInterval  = setInterval(loadStats, 30_000)

    return () => {
      clearInterval(healthInterval)
      clearInterval(statsInterval)
    }
  }, [loadStats])

  const healthLabel = {
    checking:     'Checking…',
    connected:    'API connected',
    disconnected: 'Disconnected',
  }[health]

  function fmt(n: number | undefined) {
    if (n === undefined || n === null) return '—'
    if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
    return n.toString()
  }

  return (
    <div className="app-shell">
      {/* ── Sidebar ── */}
      <aside className="sidebar" role="navigation" aria-label="Main navigation">

        {/* Logo */}
        <NavLink to="/ingest" className="sidebar-logo" aria-label="Home">
          <div className="sidebar-logo-mark">
            <BrainIcon />
          </div>
          <div className="sidebar-logo-text">
            <span className="sidebar-logo-name">Temporal RAG</span>
            <span className="sidebar-logo-sub">Belief Revision</span>
          </div>
        </NavLink>

        {/* Nav */}
        <nav className="sidebar-nav">
          {NAV.map(({ to, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
            >
              <span className="nav-icon-wrap" aria-hidden="true">{icon}</span>
              <span className="nav-label">{label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Footer: stats + health */}
        <div className="sidebar-footer">
          {/* Compact analytics metrics */}
          {stats && (
            <div className="sidebar-stats">
              <div className="stat-chip">
                <span className="stat-chip-value">{fmt(stats.total_docs)}</span>
                <span className="stat-chip-label">Docs</span>
              </div>
              <div className="stat-chip">
                <span className="stat-chip-value">{fmt(stats.total_chunks)}</span>
                <span className="stat-chip-label">Chunks</span>
              </div>
              <div className="stat-chip">
                <span className="stat-chip-value">{fmt(stats.total_queries)}</span>
                <span className="stat-chip-label">Queries</span>
              </div>
              <div className="stat-chip" style={{ borderColor: stats.unresolved_conflicts > 0 ? 'var(--warning-border)' : undefined }}>
                <span className="stat-chip-value" style={{ color: stats.unresolved_conflicts > 0 ? 'var(--warning)' : undefined }}>
                  {fmt(stats.unresolved_conflicts)}
                </span>
                <span className="stat-chip-label">Conflicts</span>
              </div>
            </div>
          )}

          {/* Health */}
          <div
            className="health-indicator"
            role="status"
            aria-label={`Backend: ${healthLabel}`}
          >
            <span className={`health-dot ${health}`} />
            <span className="health-text">
              <strong>{healthLabel}</strong>
            </span>
          </div>
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="main-content" id="main-content">
        <div className="page-wrap">
          <Outlet context={{ stats, refreshStats: loadStats } satisfies ShellContext} />
        </div>
      </main>
    </div>
  )
}
