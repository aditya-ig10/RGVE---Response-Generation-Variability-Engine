from __future__ import annotations

import os

from llama_cpp import Llama


_model: Llama | None = None


def get_model() -> Llama:
    global _model

    if _model is not None:
        return _model

    model_path = os.environ.get("RGVE_MODEL_PATH")
    if not model_path:
        raise RuntimeError("RGVE_MODEL_PATH environment variable is not set")

    _model = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=4096,
        logits_all=True,
    )
    return _model
