"""Configuration loading for ART-e evaluation."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    name: str = "art-e-eval"
    seed: int = 42
    num_samples: int = 350
    max_turns: int = 10
    output_dir: str = "outputs/eval"


@dataclass
class DatasetConfig:
    path: str = "data/art_e_vince_kaminski"
    split: str = "test"
    db_path: str = "data/enron_emails.db"


@dataclass
class JudgeConfig:
    model: str = "gpt-5-mini"
    api_key_env: str = "OPENAI_API_KEY"
    reward_correct: float = 1.0
    reward_wrong: float = -1.0
    reward_idk: float = 0.0


@dataclass
class ModelConfig:
    name: str
    backend: str  # "openai" | "anthropic" | "vllm_local"
    model_name: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    adapter_path: str | None = None
    lora_rank: int | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    max_concurrent: int = 10


@dataclass
class EvalConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    models: list[ModelConfig] = field(default_factory=list)


def load_config(yaml_path: str | Path) -> EvalConfig:
    """Load evaluation config from a YAML file."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    experiment = ExperimentConfig(**raw.get("experiment", {}))
    dataset = DatasetConfig(**raw.get("dataset", {}))
    judge = JudgeConfig(**raw.get("judge", {}))
    models = [ModelConfig(**m) for m in raw.get("models", [])]

    return EvalConfig(
        experiment=experiment,
        dataset=dataset,
        judge=judge,
        models=models,
    )
