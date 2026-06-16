from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Persona(str, Enum):
    precise = "precise"
    balanced = "balanced"
    creative = "creative"
    exploratory = "exploratory"
    expert = "expert"


class Domain(str, Enum):
    factual = "factual"
    general = "general"
    technical = "technical"
    safe = "safe"


class ParameterTensor(BaseModel):
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40)
    persona: Persona = Persona.balanced
    domain: Domain = Domain.general
    logit_bias: dict[int, float] = Field(default_factory=dict)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
