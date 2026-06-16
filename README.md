# RGVE — Response Generation Variation Explorer

Explore the possibility space of LLM outputs by sampling across parameter configurations and semantically clustering the results.

## Structure

```
rgve/
├── frontend/              # Next.js 14 (App Router) + Tailwind CSS
│   ├── app/
│   │   ├── page.tsx       # 3-column layout (parameters, responses, map)
│   │   ├── layout.tsx
│   │   └── api/           # proxied to backend
│   └── components/
│       ├── ParameterPanel.tsx
│       ├── ResponseViewer.tsx
│       └── PossibilityMap.tsx
├── backend/
│   ├── main.py            # FastAPI app — /health, /api/model/info
│   ├── models/
│   │   ├── parameter_tensor.py   # Pydantic: temperature, top_p, persona, domain, etc.
│   │   └── response_types.py     # Response, VariantBundle, PossibilityMap schemas
│   ├── engine/
│   │   ├── model_loader.py       # Singleton Llama loader (Metal, 4K ctx)
│   │   ├── generator.py          # stub
│   │   ├── explorer.py           # stub
│   │   └── clusterer.py          # stub
│   └── requirements.txt
└── README.md
```

## Status

| Component | Implemented |
|-----------|-------------|
| FastAPI app skeleton | ✅ `/health`, `/api/model/info` |
| `ParameterTensor` model | ✅ Pydantic with enums & validation |
| `Response` / `VariantBundle` schemas | ✅ Pydantic |
| `model_loader` singleton | ✅ Metal GGUF loader, `RGVE_MODEL_PATH` |
| Next.js scaffold | ✅ 3 placeholder components, Tailwind |
| Generation / exploration / clustering | ⏳ stubs |

## Getting Started

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Set the model path before calling model endpoints:

```bash
export RGVE_MODEL_PATH=/path/to/model.gguf
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:3000` and proxies `/api/*` requests to the backend at `http://127.0.0.1:8000`.

## Endpoints

```bash
# Health
curl http://localhost:8000/health
# {"status":"ok"}

# Model info (requires RGVE_MODEL_PATH)
curl http://localhost:8000/api/model/info
# {"model_name":"tinyllama_tinyllama-1.1b-chat-v1.0","context_size":4096,"gpu_layers":2147483647,"metal_active":true}
```

## Model

Tested with **TinyLlama 1.1B Chat v1.0** (Q4_K_M, 636 MB) on Apple M1 / 8 GB.

| Metric | Value |
|--------|-------|
| Prompt eval (prefill) | ~46 ms |
| Generation speed | ~55 tok/s |
| Latency | 18 ms/token |

All 23 layers offloaded to GPU via Metal. Set `RGVE_MODEL_PATH` to switch models.
