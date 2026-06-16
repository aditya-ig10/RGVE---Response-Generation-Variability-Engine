from pydantic import BaseModel
from .parameter_tensor import ParameterTensor


class Response(BaseModel):
    text: str
    parameters: ParameterTensor
    model: str


class ResponseResult(BaseModel):
    text: str
    log_prob: float
    perplexity: float
    mean_entropy: float
    token_count: int
    parameter_config: ParameterTensor


class PathResult(BaseModel):
    token_sequence: list[int]
    text: str
    log_prob: float
    entropy_profile: list[float]


class VariantBundle(BaseModel):
    prompt: str
    variants: list[Response]


class PossibilityMap(BaseModel):
    prompt: str
    top_paths: list[PathResult]
    divergent_paths: list[PathResult]
    discarded_paths: list[PathResult]
    coverage_ratio: float
    entropy_profile: list[float]
    total_nodes_expanded: int


class Cluster(BaseModel):
    label: str
    representative: Response
    members: list[Response]
