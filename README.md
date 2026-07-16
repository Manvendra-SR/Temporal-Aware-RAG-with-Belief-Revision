# Temporal-Aware RAG with Belief Revision

A research-grade Retrieval-Augmented Generation system that understands document versions, detects temporal conflicts between sources, and applies belief revision to prefer the most recent, authoritative information.

## Project Structure

```
temporal-rag/
├── .env                  # Environment variables (DATABASE_URL, etc.)
├── .env.example          # Template — copy to .env and fill in values
├── backend/              # FastAPI + SQLAlchemy + Python services
│   ├── requirements.txt
│   ├── main.py           # App entry, startup event, CORS
│   ├── config.py         # Pydantic BaseSettings from .env
│   ├── database.py       # Engine, SessionLocal, Base, get_db
│   ├── models.py         # All ORM models (Documents, Chunks, etc.)
│   └── routers/
│       └── health.py     # GET /health
└── frontend/             # React + Vite + TypeScript UI
    ├── src/
    │   ├── components/Layout.tsx   # Sidebar + health indicator
    │   └── pages/                  # 5 placeholder pages
    └── vite.config.ts    # Proxy /api + /health → :8000
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL running locally

### 1. Database Setup
```bash
psql postgres -c "CREATE DATABASE temporal_rag;"
```

### 2. Copy and configure .env
```bash
# From project root:
copy .env.example .env   # Windows
# Then edit .env with your DATABASE_URL
```

### 3. Backend
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
# source venv/bin/activate

# Install dependencies
# NOTE: On Windows, psycopg2-binary needs the --only-binary flag:
pip install psycopg2-binary --only-binary :all:
pip install fastapi==0.111.0 "uvicorn[standard]==0.29.0" sqlalchemy==2.0.30 pydantic-settings==2.2.1 python-dotenv==1.0.1

# Start the server
uvicorn main:app --reload
```

### 4. Frontend (separate terminal)
```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) — the sidebar should show a green "Connected" dot.

## API Endpoints (Phase 1)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + DB connectivity |

## Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project Skeleton | ✅ Done |
| 2 | Ingestion Pipeline | 🔜 Next |
| 3 | Hybrid Retrieval + Answer Generation | 🔜 |
| 4 | Temporal Metadata + Version Management | 🔜 |
| 5 | Temporal Reranking | 🔜 |
| 6 | Conflict Detection Engine | 🔜 |
| 7 | Belief Revision Engine | 🔜 |

## Verification

```bash
# 1. uvicorn starts without error — check terminal output
# 2. Test health endpoint
curl http://localhost:8000/health
# → {"status":"ok","db_connected":true}

# 3. Check DB tables were created
psql temporal_rag -c "\dt"
# → documents, chunks, query_log, domain_config, conflict_pairs

# 4. React app at localhost:5173 — green dot in sidebar
```
