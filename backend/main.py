from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RGVE Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/model/info")
async def model_info():
    import sys

    from engine.model_loader import get_model

    model = get_model()
    metadata = model.metadata or {}
    n_gpu = model.model_params.n_gpu_layers
    return {
        "model_name": metadata.get("general.name", "unknown"),
        "context_size": model.n_ctx(),
        "gpu_layers": n_gpu,
        "metal_active": sys.platform == "darwin" and n_gpu > 0,
    }
