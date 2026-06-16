from pydantic import BaseModel
from .parameter_tensor import ParameterTensor


class Response(BaseModel):
    text: str
    parameters: ParameterTensor
    model: str


class VariantBundle(BaseModel):
    prompt: str
    variants: list[Response]


class PossibilityMap(BaseModel):
    prompt: str
    clusters: list[Cluster]


class Cluster(BaseModel):
    label: str
    representative: Response
    members: list[Response]
