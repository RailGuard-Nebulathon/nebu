"""Small explicit trainer with AMP, clipping, checkpointing and resume."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from railguard.training.early_stopping import EarlyStopping
from railguard.training.seed import seed_everything


@dataclass
class TrainerConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    patience: int = 12
    device: str = "auto"
    seed: int = 42
    amp_dtype: str = "float16"
    matmul_precision: str = "highest"


class Trainer:
    def __init__(self, model: nn.Module, loss_fn: Callable, config: TrainerConfig | None = None, output_dir: str | Path | None = None) -> None:
        self.config = config or TrainerConfig()
        seed_everything(self.config.seed)
        selected = "cuda" if self.config.device == "auto" and torch.cuda.is_available() else ("cpu" if self.config.device == "auto" else self.config.device)
        self.device = torch.device(selected)
        if self.config.amp_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("amp_dtype must be float16, bfloat16, or float32")
        if self.config.matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError("matmul_precision must be highest, high, or medium")
        torch.set_float32_matmul_precision(self.config.matmul_precision)
        self.amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[self.config.amp_dtype]
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, nn.Module) else loss_fn
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda" and self.amp_dtype == torch.float16)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3)
        self.output_dir = Path(output_dir) if output_dir else None
        self.history: list[dict[str, float]] = []
        self.start_epoch = 0

    def _batch_loss(self, batch, training: bool) -> torch.Tensor:
        values, targets, mask, metadata = batch
        values, targets, mask = values.to(self.device), targets.to(self.device), mask.to(self.device)
        metadata = metadata.to(self.device) if metadata is not None else None
        context = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with context:
            output = self.model(values, mask=mask, metadata=metadata)
            return self.loss_fn(output, targets)

    def dry_run(self, loader) -> float:
        self.model.train()
        batch = next(iter(loader))
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._batch_loss(batch, True)
        self.scaler.scale(loss).backward()
        self.optimizer.zero_grad(set_to_none=True)
        return float(loss.detach().cpu())

    def _epoch(self, loader, training: bool) -> float:
        self.model.train(training)
        losses = []
        for batch in loader:
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                loss = self._batch_loss(batch, training)
            if training:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            losses.append(float(loss.detach().cpu()))
        return sum(losses) / max(len(losses), 1)

    def fit(self, train_loader, validation_loader=None) -> list[dict[str, float]]:
        stopping = EarlyStopping(self.config.patience)
        for epoch in range(self.start_epoch, self.config.epochs):
            train_loss = self._epoch(train_loader, True)
            validation_loss = self._epoch(validation_loader, False) if validation_loader is not None else train_loss
            self.scheduler.step(validation_loss)
            row = {"epoch": float(epoch), "train_loss": train_loss, "validation_loss": validation_loss}
            self.history.append(row)
            improved = stopping.best is None or validation_loss < stopping.best
            should_stop = stopping.update(validation_loss)
            if self.output_dir:
                self.save_checkpoint("last.pt", epoch)
                if improved:
                    self.save_checkpoint("best.pt", epoch)
            if should_stop:
                break
        return self.history

    def save_checkpoint(self, name: str, epoch: int) -> Path:
        if self.output_dir is None:
            raise ValueError("output_dir is required for checkpoints")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / name
        torch.save({"epoch": epoch, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(), "config": asdict(self.config), "history": self.history}, path)
        return path

    def resume(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.history = checkpoint.get("history", [])
        self.start_epoch = int(checkpoint["epoch"]) + 1
