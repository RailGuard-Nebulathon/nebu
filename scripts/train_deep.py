"""Explicit deep-model training/dry-run CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from railguard.config import load_config
from railguard.models.acv.relational import ACVRelationalNet
from railguard.models.corrugation.dual_domain import CorrugationNet
from railguard.models.door.detector import DoorNet
from railguard.models.shm.damage_regressor import DamageNet
from railguard.training.datasets import SequenceDataset, pad_collate
from railguard.training.losses import gaussian_nll
from railguard.training.trainer import Trainer, TrainerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("door", "acv", "corrugation", "shm"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use deterministic architecture-smoke tensors")
    parser.add_argument("--preprocessed", type=Path, help="NPZ containing arrays x, y and optional mask/metadata/car_mask")
    parser.add_argument("--max-samples", type=int, help="Optional lightweight subset; omit for full preprocessed data")
    parser.add_argument("--epochs", type=int, help="Overrides training.epochs from YAML")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Overrides runtime.device from YAML")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoints/deep"))
    args = parser.parse_args()
    config = load_config(args.config) if args.config else load_config()
    selected_device = args.device or config.runtime.device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    epochs = args.epochs or config.training.epochs
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[config.runtime.amp_dtype]
    torch.set_float32_matmul_precision(config.runtime.matmul_precision)
    synthetic_mode = args.synthetic or (args.dry_run and args.preprocessed is None)
    if not synthetic_mode and args.preprocessed is None:
        raise SystemExit("Real deep training requires --preprocessed NPZ from an inspected, leakage-safe training split; use --dry-run for a synthetic architecture smoke test.")
    generator = torch.Generator().manual_seed(args.seed)
    smoke_samples = args.max_samples or 8
    if synthetic_mode:
        print("Using deterministic synthetic tensors for architecture smoke testing only.")
    if args.preprocessed:
        arrays = np.load(args.preprocessed)
        tensor_x, tensor_y = torch.from_numpy(arrays["x"]).float(), torch.from_numpy(arrays["y"])
        loaded_metadata = torch.from_numpy(arrays["metadata"]) if "metadata" in arrays else None
        loaded_car_mask = torch.from_numpy(arrays["car_mask"]).bool() if "car_mask" in arrays else None
        if tensor_x.ndim == 4 and loaded_car_mask is not None:
            valid_values = tensor_x[loaded_car_mask]
            channel_mean = valid_values.mean(dim=(0, 1), keepdim=True).unsqueeze(0)
            channel_std = valid_values.std(dim=(0, 1), keepdim=True).clamp_min(1e-6).unsqueeze(0)
        else:
            reduce_axes = tuple(range(tensor_x.ndim - 1))
            channel_mean = tensor_x.mean(dim=reduce_axes, keepdim=True)
            channel_std = tensor_x.std(dim=reduce_axes, keepdim=True).clamp_min(1e-6)
        tensor_x = (tensor_x - channel_mean) / channel_std
        if tensor_x.ndim == 4 and loaded_car_mask is not None:
            tensor_x = tensor_x.masked_fill(~loaded_car_mask[:, :, None, None], 0.0)
        selected_x = tensor_x[: args.max_samples] if args.max_samples else tensor_x
        sequences = [item for item in selected_x]
    else:
        tensor_y = None
        loaded_metadata = None
        loaded_car_mask = None
        sequences = [torch.randn(64 + i % 3, 6, generator=generator) for i in range(smoke_samples)]
    metadata = None
    input_channels = int(sequences[0].shape[-1])
    if args.task == "door":
        class_weights = None
        if tensor_y is not None:
            selected_y = tensor_y[: args.max_samples] if args.max_samples else tensor_y
            counts = torch.bincount(selected_y.long(), minlength=2).float().clamp_min(1)
            class_weights = counts.sum() / (len(counts) * counts)
        model, loss = DoorNet(input_channels, 16), nn.CrossEntropyLoss(weight=class_weights)
        targets = [tensor_y[i].long() if tensor_y is not None else torch.tensor(i % 2) for i in range(len(sequences))]
        metadata = [loaded_metadata[i].long() if loaded_metadata is not None else torch.tensor(i % 2) for i in range(len(sequences))]
    elif args.task == "corrugation":
        class_weights = None
        if tensor_y is not None:
            selected_y = tensor_y[: args.max_samples] if args.max_samples else tensor_y
            counts = torch.bincount(selected_y.long(), minlength=3).float().clamp_min(1)
            class_weights = counts.sum() / (len(counts) * counts)
        model, loss = CorrugationNet(input_channels, 16), nn.CrossEntropyLoss(weight=class_weights)
        targets = [tensor_y[i].long() if tensor_y is not None else torch.tensor(i % 3) for i in range(len(sequences))]
    elif args.task == "shm":
        model, loss = DamageNet(input_channels, heteroscedastic=True, hidden_channels=16), gaussian_nll
        targets = [tensor_y[i].float() if tensor_y is not None else torch.tensor(float(i) / len(sequences)) for i in range(len(sequences))]
    else:
        cars, time, channels = (tensor_x.shape[1:] if args.preprocessed else (8, 64, 6))
        device = torch.device(selected_device)
        model = ACVRelationalNet(channels, 16, heads=4).to(device)
        x = (tensor_x[: args.max_samples] if args.max_samples else tensor_x) if args.preprocessed else torch.randn(smoke_samples, cars, time, channels, generator=generator)
        y = ((tensor_y[: args.max_samples] if args.max_samples else tensor_y).long() if tensor_y is not None else torch.arange(smoke_samples) % cars)
        x, y = x.to(device), y.to(device)
        if loaded_car_mask is not None:
            car_mask = loaded_car_mask[: args.max_samples] if args.max_samples else loaded_car_mask
            car_mask = car_mask.to(device)
        else:
            car_mask = torch.ones(len(x), cars, dtype=torch.bool, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
        model.train(); optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda" and amp_dtype != torch.float32):
            logits = model(x, car_mask=car_mask); loss = nn.functional.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        if args.dry_run:
            print(f"device={device}; amp_dtype={config.runtime.amp_dtype}; dry_run_loss={float(loss.detach()):.6f}"); return
        for _ in range(epochs):
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda" and amp_dtype != torch.float32):
                logits=model(x,car_mask=car_mask); loss=nn.functional.cross_entropy(logits,y)
            scaler.scale(loss).backward()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.preprocessed:
            np.savez(args.output_dir / "normalization.npz", mean=channel_mean.cpu().numpy(), std=channel_std.cpu().numpy())
        torch.save({"model":model.state_dict(),"task":"acv"},args.output_dir/"last.pt"); print(args.output_dir/"last.pt"); return
    loader = DataLoader(SequenceDataset(sequences, targets, metadata), batch_size=config.training.batch_size, collate_fn=pad_collate)
    trainer = Trainer(
        model,
        loss,
        TrainerConfig(
            epochs=epochs,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            patience=config.training.early_stopping_patience,
            device=selected_device,
            seed=args.seed,
            amp_dtype=config.runtime.amp_dtype,
            matmul_precision=config.runtime.matmul_precision,
        ),
        args.output_dir,
    )
    if args.preprocessed and not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(args.output_dir / "normalization.npz", mean=channel_mean.cpu().numpy(), std=channel_std.cpu().numpy())
    if args.resume:
        trainer.resume(args.resume)
    if args.dry_run:
        print(f"device={trainer.device}; amp_dtype={config.runtime.amp_dtype}; dry_run_loss={trainer.dry_run(loader):.6f}")
    else:
        print(trainer.fit(loader))


if __name__ == "__main__":
    main()
