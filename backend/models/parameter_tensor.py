from dataclasses import dataclass


@dataclass
class ParameterTensor:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    typical_p: float = 1.0
    mirostat_mode: int = 0
    seed: int | None = None
