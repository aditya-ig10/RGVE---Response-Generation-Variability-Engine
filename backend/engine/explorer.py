from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from models.response_types import PathResult, PossibilityMap


@dataclass(order=True)
class _FrontierNode:
    neg_log_prob: float
    _id: int
    node: "TreeNode" = field(compare=False)


@dataclass
class TreeNode:
    token_ids: list[int]
    log_prob: float
    depth: int
    step_entropies: list[float]


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits, dtype=np.float64)
    return exp / np.sum(exp)


def _step_entropy(probs: np.ndarray) -> float:
    clipped = probs[probs > 1e-12]
    return float(-np.sum(clipped * np.log(clipped)))


def _nucleus_filter(
    probs: np.ndarray, top_p: float, max_tokens: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumsum = np.cumsum(sorted_probs, dtype=np.float64)
    cutoff = int(np.searchsorted(cumsum, top_p)) + 1
    cutoff = min(cutoff, max_tokens)
    keep = sorted_indices[:cutoff]
    pruned = sorted_indices[cutoff:]
    return keep, pruned, probs[keep], probs[pruned]


def _path_result(
    token_ids: list[int],
    log_prob: float,
    entropy_profile: list[float],
    detokenize_fn,
) -> PathResult:
    text = detokenize_fn(token_ids).decode("utf-8", errors="replace")
    return PathResult(
        token_sequence=token_ids,
        text=text,
        log_prob=log_prob,
        entropy_profile=entropy_profile,
    )


def explore_possibility_space(
    prompt: str, budget: int = 3, top_p: float = 0.95, max_branch: int = 3, max_depth: int = 8
) -> PossibilityMap:
    from engine.model_loader import get_model

    model = get_model()

    prompt_bytes = prompt.encode("utf-8")
    prompt_tokens = model.tokenize(prompt_bytes)

    def _detok(ids: list[int]) -> bytes:
        return model.detokenize(ids)

    frontier: list[_FrontierNode] = []
    explored: list[PathResult] = []
    discarded: list[PathResult] = []
    discarded_mass = 0.0
    discard_sample: list[tuple[float, int, PathResult]] = []
    node_id = 0

    model.reset()
    model.eval(prompt_tokens)
    logits = model._scores[-1].astype(np.float64)
    probs = _softmax(logits)
    entropy = _step_entropy(probs)

    keep_ids, prune_ids, keep_probs, prune_probs = _nucleus_filter(probs, top_p, max_branch)

    for token, prob in zip(keep_ids, keep_probs):
        lp = math.log(float(prob))
        child = TreeNode(
            token_ids=[int(token)],
            log_prob=lp,
            depth=1,
            step_entropies=[entropy],
        )
        heapq.heappush(frontier, _FrontierNode(-lp, node_id, child))
        node_id += 1

    for token, prob in zip(prune_ids, prune_probs):
        lp = math.log(float(prob))
        discarded_mass += math.exp(lp)
        if len(discard_sample) < max_branch * 5:
            pr = _path_result([int(token)], lp, [entropy], _detok)
            heapq.heappush(discard_sample, (-lp, node_id, pr))

    eos_id = model.token_eos()
    max_depth = max_depth

    while len(explored) < budget and frontier:
        entry = heapq.heappop(frontier)
        node = entry.node

        full_tokens = prompt_tokens + node.token_ids

        if node.token_ids[-1] == eos_id:
            pr = _path_result(
                node.token_ids, node.log_prob, node.step_entropies, _detok
            )
            explored.append(pr)
            continue

        if node.depth >= max_depth:
            pr = _path_result(
                node.token_ids, node.log_prob, node.step_entropies, _detok
            )
            explored.append(pr)
            continue

        model.reset()
        model.eval(full_tokens)
        next_logits = model._scores[-1].astype(np.float64)
        next_probs = _softmax(next_logits)
        step_ent = _step_entropy(next_probs)

        keep_ids, prune_ids, keep_probs, prune_probs = _nucleus_filter(
            next_probs, top_p, max_branch
        )

        for token, prob in zip(keep_ids, keep_probs):
            lp = node.log_prob + math.log(float(prob))
            child = TreeNode(
                token_ids=node.token_ids + [int(token)],
                log_prob=lp,
                depth=node.depth + 1,
                step_entropies=node.step_entropies + [step_ent],
            )
            heapq.heappush(frontier, _FrontierNode(-lp, node_id, child))
            node_id += 1

        node_prob = math.exp(node.log_prob)
        prune_sum = float(np.sum(prune_probs))
        discarded_mass += node_prob * prune_sum

        for token, prob in zip(prune_ids, prune_probs):
            lp = node.log_prob + math.log(float(prob))
            if len(discard_sample) < max_branch * 5:
                pr = _path_result(
                    node.token_ids + [int(token)], lp, node.step_entropies + [step_ent], _detok
                )
                heapq.heappush(discard_sample, (-lp, node_id, pr))
                node_id += 1

    explored_mass = sum(math.exp(pr.log_prob) for pr in explored)
    total_mass = explored_mass + discarded_mass
    coverage = explored_mass / total_mass if total_mass > 0 else 0.0

    discarded_top = sorted(
        (pr for _, _, pr in discard_sample),
        key=lambda x: x.log_prob,
        reverse=True,
    )[:10]

    top_paths = sorted(explored, key=lambda x: x.log_prob, reverse=True)[:5]
    divergent_paths = sorted(explored, key=lambda x: x.log_prob)[:5]

    max_d = max((len(p.entropy_profile) for p in explored), default=0)
    entropy_profile: list[float] = []
    for i in range(max_d):
        vals = [p.entropy_profile[i] for p in explored if i < len(p.entropy_profile)]
        if vals:
            entropy_profile.append(float(np.mean(vals)))

    return PossibilityMap(
        prompt=prompt,
        top_paths=top_paths,
        divergent_paths=divergent_paths,
        discarded_paths=discarded_top,
        coverage_ratio=coverage,
        entropy_profile=entropy_profile,
        total_nodes_expanded=node_id,
    )
