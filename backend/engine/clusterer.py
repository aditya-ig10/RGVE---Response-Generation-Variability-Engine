from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from models.parameter_tensor import ParameterTensor, Persona, Domain
from models.response_types import ResponseResult, ScoredResponse, ClusterResult, VariantBundle

PARAMETER_MANIFEST: list[ParameterTensor] = [
    ParameterTensor(temperature=0.1, top_p=0.9, top_k=40, persona=Persona.precise, domain=Domain.factual, repetition_penalty=1.1),
    ParameterTensor(temperature=0.5, top_p=0.9, top_k=40, persona=Persona.balanced, domain=Domain.general, repetition_penalty=1.1),
    ParameterTensor(temperature=0.8, top_p=0.9, top_k=40, persona=Persona.creative, domain=Domain.general, repetition_penalty=1.1),
    ParameterTensor(temperature=1.2, top_p=0.95, top_k=50, persona=Persona.exploratory, domain=Domain.general, repetition_penalty=1.0),
    ParameterTensor(temperature=0.4, top_p=0.9, top_k=40, persona=Persona.expert, domain=Domain.technical, repetition_penalty=1.1),
    ParameterTensor(temperature=0.7, top_p=0.9, top_k=40, persona=Persona.balanced, domain=Domain.safe, repetition_penalty=1.1),
]


_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def encode_texts(texts: list[str]) -> np.ndarray:
    return _get_encoder().encode(texts, show_progress_bar=False)


def cluster_embeddings(
    embeddings: np.ndarray,
) -> tuple[list[ClusterResult], float]:
    n = len(embeddings)
    if n < 2:
        return [ClusterResult(cluster_id=0, member_indices=[0], size=1)], 0.0

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.3,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(i)

    results = [
        ClusterResult(cluster_id=cid, member_indices=members, size=len(members))
        for cid, members in clusters.items()
    ]

    inter = _inter_cluster_distance(embeddings, labels)
    return results, inter


def _inter_cluster_distance(embeddings: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    dists = []
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            mask_i = labels == unique[i]
            mask_j = labels == unique[j]
            centroids = np.mean(embeddings[mask_i], axis=0), np.mean(embeddings[mask_j], axis=0)
            d = cosine_distances([centroids[0]], [centroids[1]])[0, 0]
            dists.append(d)
    return float(np.mean(dists)) if dists else 0.0


def project_umap(embeddings: np.ndarray) -> np.ndarray:
    import umap

    n = len(embeddings)
    if n < 3:
        coords = np.zeros((n, 2))
        if n == 2:
            coords[1, 0] = 1.0
        return coords

    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    return reducer.fit_transform(embeddings)


def score_responses(
    responses: list[ResponseResult],
    embeddings: np.ndarray,
) -> list[ScoredResponse]:
    n = len(responses)
    scored: list[ScoredResponse] = []

    for i, r in enumerate(responses):
        quality = -r.perplexity
        uncertainty = r.mean_entropy

        dists = cosine_distances([embeddings[i]], embeddings)[0]
        diversity = float(np.mean([d for j, d in enumerate(dists) if j != i]))

        scored.append(ScoredResponse(
            response=r,
            quality=quality,
            diversity=diversity,
            uncertainty=uncertainty,
        ))

    return scored


LABELS = [
    "precise/factual t=0.1",
    "balanced/general t=0.5",
    "creative/general t=0.8",
    "exploratory/general t=1.2",
    "expert/technical t=0.4",
    "balanced/safe t=0.7",
]


def build_variant_bundle(
    prompt: str,
    responses: list[ResponseResult],
) -> VariantBundle:
    texts = [r.text for r in responses]
    embeddings = encode_texts(texts)

    clusters, inter_dist = cluster_embeddings(embeddings)
    umap_2d = project_umap(embeddings)

    scored = score_responses(responses, embeddings)
    scored.sort(key=lambda s: s.quality, reverse=True)
    primary = scored[0]
    alternatives = scored[1:]

    umap_coords = []
    for idx, r in enumerate(responses):
        cid = _cluster_id_for_index(clusters, idx)
        umap_coords.append({
            "text": r.text[:80],
            "x": float(umap_2d[idx, 0]),
            "y": float(umap_2d[idx, 1]),
            "cluster_id": cid,
            "config_label": LABELS[idx] if idx < len(LABELS) else f"variant_{idx}",
        })

    return VariantBundle(
        primary=primary,
        alternatives=alternatives,
        semantic_clusters=clusters,
        inter_cluster_distance=inter_dist,
        umap_coords=umap_coords,
    )


def _cluster_id_for_index(clusters: list[ClusterResult], idx: int) -> int:
    for c in clusters:
        if idx in c.member_indices:
            return c.cluster_id
    return -1
