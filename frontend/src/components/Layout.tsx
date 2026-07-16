import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import axios from 'axios'

type HealthStatus = 'checking' | 'connected' | 'disconnected'

interface NavItem {
  to: string
  icon: string
  label: string
  phase: string
}

const navItems: NavItem[] = [
  { to: '/',          icon: '⌂',  label: 'Overview',   phase: 'Phase 1' },
  { to: '/ingest',    icon: '↑',  label: 'Ingest',     phase: 'Phase 2' },
  { to: '/documents', icon: '⊞',  label: 'Documents',  phase: 'Phase 2' },
  { to: '/query',     icon: '⚡', label: 'Query',      phase: 'Phase 3' },
  { to: '/conflicts', icon: '⚠',  label: 'Conflicts',  phase: 'Phase 6' },
]

export default function Layout() {
  const [health, setHealth] = useState<HealthStatus>('checking')
  const [dbConnected, setDbConnected] = useState(false)

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const { data } = await axios.get('/health', { timeout: 4000 })
        setDbConnected(data.db_connected)
        setHealth(data.db_connected ? 'connected' : 'disconnected')
      } catch {
        setHealth('disconnected')
        setDbConnected(false)
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 5000)
    return () => clearInterval(interval)
  }, [])

  const healthLabel = {
    checking: 'Checking…',
    connected: 'Connected',
    disconnected: 'Disconnected',
  }[health]

  return (
    <div className="app-shell">
      {/* ── Sidebar ── */}
      <aside className="sidebar" role="navigation" aria-label="Main navigation">
        {/* Logo */}
        <div className="sidebar-header">
          <NavLink to="/" className="sidebar-logo">
            <div className="sidebar-logo-icon">🧠</div>
            <span className="sidebar-logo-text">
              Temporal RAG
              <span className="sidebar-logo-sub">Belief Revision</span>
            </span>
          </NavLink>
        </div>

        {/* Nav links */}
        <nav className="sidebar-nav">
          <span className="nav-section-label">Navigation</span>
          {navItems.map(({ to, icon, label, phase }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
              aria-current={undefined}
            >
              <span className="nav-icon" aria-hidden="true">{icon}</span>
              {label}
              <span className="nav-badge">{phase}</span>
            </NavLink>
          ))}
        </nav>

        {/* Health indicator */}
        <div className="sidebar-footer">
          <div
            className="health-indicator"
            role="status"
            aria-label={`Backend status: ${healthLabel}`}
            title={`Database: ${dbConnected ? 'connected' : 'disconnected'}`}
          >
            <span className={`health-dot ${health}`} />
            <span className="health-text">
              <strong>{healthLabel}</strong>
              {health === 'connected' ? 'API · DB online' : 'Backend unreachable'}
            </span>
          </div>
        </div>
      </aside>

      {/* ── Main content ── */}
      <main className="main-content" id="main-content">
        <Outlet />
      </main>
    </div>
  )
}
