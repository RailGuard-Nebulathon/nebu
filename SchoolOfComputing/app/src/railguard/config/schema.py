"""Validated application configuration."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProjectConfig(BaseModel):
    name: str = "railguard"
    seed: int = 42


class PathsConfig(BaseModel):
    data_root: Path = Path("data")
    raw_root: Path = Path("data/raw")
    processed_root: Path = Path("data/processed")
    manifest_root: Path = Path("data/manifests")
    output_root: Path = Path("outputs")


class RuntimeConfig(BaseModel):
    device: str = "auto"
    num_workers: int = Field(default=0, ge=0)
    deterministic: bool = True
    amp_dtype: str = "float16"
    matmul_precision: str = "highest"


class TrainingConfig(BaseModel):
    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=32, gt=0)
    early_stopping_patience: int = Field(default=12, ge=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    project: ProjectConfig = ProjectConfig()
    paths: PathsConfig = PathsConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    training: TrainingConfig = TrainingConfig()
    task: str | None = None
    model: str | None = None
    logging: dict[str, Any] = {"level": "INFO"}
    evaluation: dict[str, Any] = {}
    robustness: dict[str, Any] = {}
