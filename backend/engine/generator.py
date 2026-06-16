from __future__ import annotations

import math

from models.parameter_tensor import ParameterTensor, Persona
from models.response_types import ResponseResult

PERSONA_PREFIXES: dict[Persona, str] = {
    Persona.precise: "Answer precisely and factually.",
    Persona.balanced: "Provide a balanced and well-reasoned answer.",
    Persona.creative: "Answer imaginatively and explore ideas.",
    Persona.exploratory: "Consider many perspectives and explore possibilities.",
    Persona.expert: "Answer as a domain expert with technical depth.",
}


def _build_prompt(raw: str, persona: Persona) -> str:
    prefix = PERSONA_PREFIXES[persona]
    return f"{prefix}\n\n{raw}"


def generate_response(
    prompt: str,
    theta: ParameterTensor,
) -> ResponseResult:
    from engine.model_loader import get_model

    model = get_model()

    full_prompt = _build_prompt(prompt, theta.persona)

    kwargs = dict(
        prompt=full_prompt,
        temperature=theta.temperature,
        top_p=theta.top_p,
        top_k=theta.top_k,
        repeat_penalty=theta.repetition_penalty,
        logprobs=5,
    )
    if theta.logit_bias:
        kwargs["logit_bias"] = theta.logit_bias

    output = model.create_completion(**kwargs)

    choice = output["choices"][0]
    text = choice["text"]
    logprobs_data = choice.get("logprobs")

    if logprobs_data and logprobs_data.get("token_logprobs"):
        step_logprobs = logprobs_data["token_logprobs"]
        top_logprobs_list = logprobs_data.get("top_logprobs", [])

        total_log_prob = sum(step_logprobs)
        mean_log_prob = total_log_prob / len(step_logprobs)
        perplexity = math.exp(-mean_log_prob)

        entropies = []
        for top_dict in top_logprobs_list:
            if top_dict is None:
                continue
            probs = [math.exp(lp) for lp in top_dict.values()]
            total_p = sum(probs)
            if total_p > 0:
                normed = [p / total_p for p in probs]
                entropies.append(-sum(p * math.log(p) for p in normed))
        mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    else:
        total_log_prob = 0.0
        perplexity = 0.0
        mean_entropy = 0.0

    return ResponseResult(
        text=text,
        log_prob=total_log_prob,
        perplexity=perplexity,
        mean_entropy=mean_entropy,
        token_count=output["usage"]["completion_tokens"],
        parameter_config=theta,
    )
