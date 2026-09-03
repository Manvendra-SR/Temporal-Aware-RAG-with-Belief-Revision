import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import IngestPage from './pages/IngestPage'
import LibraryPage from './pages/LibraryPage'
import QueryPage from './pages/QueryPage'
import ConflictsPage from './pages/ConflictsPage'

/**
 * Four pages, matching the four steps of the workflow:
 *   Ingest    — add a document or a new version of one
 *   Library   — see what is loaded and how versions relate
 *   Query     — ask a question and inspect how it was answered
 *   Conflicts — review contradictions the system found
 *
 * `/documents` is kept as a redirect because it was a real URL before the
 * Documents and Timeline pages were merged into Library as tabs.
 */
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/query" replace />} />
        <Route path="ingest"    element={<IngestPage />} />
        <Route path="library"   element={<LibraryPage />} />
        <Route path="query"     element={<QueryPage />} />
        <Route path="conflicts" element={<ConflictsPage />} />
        <Route path="documents" element={<Navigate to="/library" replace />} />
        <Route path="*"         element={<Navigate to="/query" replace />} />
      </Route>
    </Routes>
  )
}
