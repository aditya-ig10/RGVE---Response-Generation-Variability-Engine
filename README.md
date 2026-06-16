# RGVE — Response Generation Variation Engine

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
│   ├── main.py            # FastAPI app — /health, /api/model/info, /api/generate, /api/explore, /api/variants
│   ├── models/
│   │   ├── parameter_tensor.py   # Pydantic: temperature, top_p, persona, domain, etc.
│   │   └── response_types.py     # ResponseResult, PathResult, ScoredResponse, VariantBundle, PossibilityMap
│   ├── engine/
│   │   ├── model_loader.py       # Singleton Llama loader (Metal, 4K ctx, logits_all)
│   │   ├── generator.py          # generate_response() with entropy/perplexity
│   │   ├── explorer.py           # explore_possibility_space() priority-queue tree search
│   │   └── clusterer.py          # PARAMETER_MANIFEST, sentence-transformers encoding, agglomerative clustering, UMAP, quality scoring
│   └── requirements.txt
└── README.md
```

## Status

| Component | Implemented |
|-----------|-------------|
| FastAPI app skeleton | ✅ `/health`, `/api/model/info` |
| `ParameterTensor` model | ✅ Pydantic with enums & validation |
| `ResponseResult`/`PathResult` schemas | ✅ Pydantic |
| `model_loader` singleton | ✅ Metal GGUF loader, `RGVE_MODEL_PATH`, `logits_all=True` |
| `generator.py` | ✅ `generate_response()` with entropy/perplexity, persona prefixes |
| `explorer.py` | ✅ nucleus-filtered priority-queue tree search with coverage |
| Next.js scaffold | ✅ 3 placeholder components, Tailwind |
| `clusterer.py` | ✅ 6-config PARAMETER_MANIFEST, encoding, clustering, UMAP, scoring, SSE streaming |

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

# Generate single response
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is the capital of France?","theta":{"temperature":0.7,"top_p":0.9,"top_k":40,"persona":"precise","domain":"general","logit_bias":{},"repetition_penalty":1.0}}'
# {"text":"\n\nAnswer: Paris.","log_prob":...,"perplexity":1.82,"mean_entropy":...,"token_count":6,...}

# Explore possibility space (tree search)
curl -X POST http://localhost:8000/api/explore \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a short poem about","budget":5,"top_p":0.92,"max_branch":4,"max_depth":6}'
# {"prompt":"...","top_paths":[...],"coverage_ratio":0.0236,"total_nodes_expanded":180,...}

# Generate variants across 6 parameter configs (SSE stream)
curl -N -X POST http://localhost:8000/api/variants \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Tell me about machine learning"}'
# event: variant   data: {"index":0,"label":"precise/factual t=0.1","text":"..."}
# event: variant   data: {"index":1,"label":"balanced/general t=0.5","text":"..."}
# ...
# event: complete  data: {"primary":{...},"alternatives":[...],"semantic_clusters":[...],"umap_coords":[...]}
```

## Model

Tested with **TinyLlama 1.1B Chat v1.0** (Q4_K_M, 636 MB) on Apple M1.

| Metric | Value |
|--------|-------|
| Prompt eval (prefill) | ~46 ms |
| Generation speed | ~55 tok/s |
| Latency | 18 ms/token |
| Generate (single) | ~0.3 s |
| Explore (budget=3, branch=3, depth=6) | ~2.2 s |
| Explore (budget=5, branch=4, depth=6) | ~3.0 s |
| Variants (6 configs + encoding + UMAP) | ~3.5 s |

All 23 layers offloaded to GPU via Metal. Set `RGVE_MODEL_PATH` to switch models.

### Performance Notes

- **First request after server start** takes 10–20 s due to Metal shader compilation (cached thereafter).
- **Explore speed** is bounded by `budget × max_branch × max_depth × 50 ms` — each tree node requires a full `model.reset() + model.eval()` since the KV cache cannot be shared across branches. Conservative defaults keep responses under 3 s.
- **Variants** streams via SSE: 6 `variant` events as each config finishes, then a `complete` event with the full `VariantBundle` (clusters, UMAP coords, scored responses).
- Models without `logits_all=True` support will not return logprobs, entropy, or explore results.
