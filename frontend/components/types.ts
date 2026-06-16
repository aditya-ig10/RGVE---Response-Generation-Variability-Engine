export type Persona = "precise" | "balanced" | "creative" | "exploratory" | "expert";
export type Domain = "factual" | "general" | "technical" | "safe";

export interface ResponseResult {
  text: string;
  log_prob: number;
  perplexity: number;
  mean_entropy: number;
  token_count: number;
  parameter_config: ThetaParams;
}

export interface ThetaParams {
  temperature: number;
  top_p: number;
  top_k: number;
  persona: Persona;
  domain: Domain;
  logit_bias: Record<string, number>;
  repetition_penalty: number;
}

export interface ScoredResponse {
  response: ResponseResult;
  quality: number;
  diversity: number;
  uncertainty: number;
}

export interface ClusterResult {
  cluster_id: number;
  member_indices: number[];
  size: number;
}

export interface UmapCoord {
  text: string;
  x: number;
  y: number;
  cluster_id: number;
  config_label: string;
}

export interface VariantBundle {
  primary: ScoredResponse;
  alternatives: ScoredResponse[];
  semantic_clusters: ClusterResult[];
  inter_cluster_distance: number;
  umap_coords: UmapCoord[];
}

export interface PathResult {
  token_sequence: number[];
  text: string;
  log_prob: number;
  entropy_profile: number[];
}

export interface PossibilityMap {
  prompt: string;
  top_paths: PathResult[];
  divergent_paths: PathResult[];
  discarded_paths: PathResult[];
  coverage_ratio: number;
  entropy_profile: number[];
  total_nodes_expanded: number;
}
