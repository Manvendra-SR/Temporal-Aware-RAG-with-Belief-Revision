import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import HomePage from './pages/HomePage'
import IngestPage from './pages/IngestPage'
import DocumentsPage from './pages/DocumentsPage'
import QueryPage from './pages/QueryPage'
import ConflictsPage from './pages/ConflictsPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="ingest" element={<IngestPage />} />
        <Route path="documents" element={<DocumentsPage />} />
        <Route path="query" element={<QueryPage />} />
        <Route path="conflicts" element={<ConflictsPage />} />
      </Route>
    </Routes>
  )
}
