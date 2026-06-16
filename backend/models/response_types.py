from pydantic import BaseModel
from .parameter_tensor import ParameterTensor


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


class ScoredResponse(BaseModel):
    response: ResponseResult
    quality: float
    diversity: float
    uncertainty: float


class ClusterResult(BaseModel):
    cluster_id: int
    member_indices: list[int]
    size: int


class VariantBundle(BaseModel):
    primary: ScoredResponse
    alternatives: list[ScoredResponse]
    semantic_clusters: list[ClusterResult]
    inter_cluster_distance: float
    umap_coords: list[dict]


class PossibilityMap(BaseModel):
    prompt: str
    top_paths: list[PathResult]
    divergent_paths: list[PathResult]
    discarded_paths: list[PathResult]
    coverage_ratio: float
    entropy_profile: list[float]
    total_nodes_expanded: int
