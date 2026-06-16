from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.parameter_tensor import ParameterTensor

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


class GenerateRequest(BaseModel):
    prompt: str
    theta: ParameterTensor


@app.post("/api/generate")
async def generate(body: GenerateRequest):
    from engine.generator import generate_response

    result = generate_response(body.prompt, body.theta)
    return result


class ExploreRequest(BaseModel):
    prompt: str
    budget: int = 100
    top_p: float = 0.95
    max_branch: int = 10
    max_depth: int = 15


@app.post("/api/explore")
async def explore(body: ExploreRequest):
    from engine.explorer import explore_possibility_space

    result = explore_possibility_space(body.prompt, body.budget, body.top_p, body.max_branch, body.max_depth)
    return result
