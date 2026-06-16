# RGVE — Response Generation Variation Engine

Explore the possibility space of LLM outputs by sampling across parameter configurations and semantically clustering the results.

## Structure

```
rgve/
├── frontend/          # Next.js 14 (App Router) + Tailwind CSS
│   ├── app/
│   │   ├── page.tsx
│   │   ├── layout.tsx
│   │   └── api/              # proxied to backend
│   └── components/
│       ├── ParameterPanel.tsx
│       ├── ResponseViewer.tsx
│       └── PossibilityMap.tsx
├── backend/
│   ├── main.py               # FastAPI app
│   ├── models/               # ParameterTensor, Response schemas
│   ├── engine/               # generation, exploration, clustering
│   └── requirements.txt
└── README.md
```

## Getting Started

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:3000` and proxies `/api/*` requests to the backend at `http://127.0.0.1:8000`.

## Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```
