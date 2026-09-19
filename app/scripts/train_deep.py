"""Explicit deep-model training/dry-run CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from railguard.config import load_config
from railguard.models.acv.relational import ACVRelationalNet
from railguard.models.corrugation.dual_domain import CorrugationNet
from railguard.models.door.detector import DoorNet
from railguard.models.shm.damage_regressor import DamageNet
from railguard.training.datasets import SequenceDataset, pad_collate
from railguard.training.losses import smooth_mape
from railguard.training.trainer import Trainer, TrainerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("door", "acv", "corrugation", "shm"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--synthetic", action="store_true", help="Use deterministic architecture-smoke tensors"
    )
    parser.add_argument(
        "--preprocessed",
        type=Path,
        help="NPZ containing arrays x, y and optional mask/metadata/car_mask",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional lightweight subset; omit for full preprocessed data",
    )
    parser.add_argument("--epochs", type=int, help="Overrides training.epochs from YAML")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), help="Overrides runtime.device from YAML"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    args.output_dir = args.output_dir or Path("outputs/checkpoints/deep") / args.task
    config = load_config(args.config) if args.config else load_config()
    selected_device = args.device or config.runtime.device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    epochs = args.epochs or config.training.epochs
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        config.runtime.amp_dtype
    ]
    torch.set_float32_matmul_precision(config.runtime.matmul_precision)
    synthetic_mode = args.synthetic or (args.dry_run and args.preprocessed is None)
    if not synthetic_mode and args.preprocessed is None:
        raise SystemExit(
            "Real deep training requires --preprocessed NPZ from an inspected, leakage-safe training split; use --dry-run for a synthetic architecture smoke test."
        )
    generator = torch.Generator().manual_seed(args.seed)
    smoke_samples = args.max_samples or 8
    if synthetic_mode:
        print("Using deterministic synthetic tensors for architecture smoke testing only.")
    if args.preprocessed:
        arrays = np.load(args.preprocessed)
        tensor_x, tensor_y = torch.from_numpy(arrays["x"]).float(), torch.from_numpy(arrays["y"])
        loaded_metadata = torch.from_numpy(arrays["metadata"]) if "metadata" in arrays else None
        loaded_car_mask = (
            torch.from_numpy(arrays["car_mask"]).bool() if "car_mask" in arrays else None
        )
        if args.max_samples:
            tensor_x, tensor_y = tensor_x[: args.max_samples], tensor_y[: args.max_samples]
            loaded_metadata = (
                loaded_metadata[: args.max_samples] if loaded_metadata is not None else None
            )
            loaded_car_mask = (
                loaded_car_mask[: args.max_samples] if loaded_car_mask is not None else None
            )
        all_indices = np.arange(len(tensor_x))
        stratify = tensor_y.numpy() if args.task in {"door", "corrugation"} else None
        if stratify is not None and np.min(np.unique(stratify, return_counts=True)[1]) < 2:
            stratify = None
        train_indices, validation_indices = train_test_split(
            all_indices,
            test_size=max(1, int(round(0.2 * len(all_indices)))),
            random_state=args.seed,
            stratify=stratify,
        )
        train_tensor = tensor_x[train_indices]
        if tensor_x.ndim == 4 and loaded_car_mask is not None:
            valid_values = train_tensor[loaded_car_mask[train_indices]]
            channel_mean = valid_values.mean(dim=(0, 1), keepdim=True).unsqueeze(0)
            channel_std = valid_values.std(dim=(0, 1), keepdim=True).clamp_min(1e-6).unsqueeze(0)
        else:
            reduce_axes = tuple(range(train_tensor.ndim - 1))
            channel_mean = train_tensor.mean(dim=reduce_axes, keepdim=True)
            channel_std = train_tensor.std(dim=reduce_axes, keepdim=True).clamp_min(1e-6)
        tensor_x = (tensor_x - channel_mean) / channel_std
        if tensor_x.ndim == 4 and loaded_car_mask is not None:
            tensor_x = tensor_x.masked_fill(~loaded_car_mask[:, :, None, None], 0.0)
        sequences = [item for item in tensor_x]
    else:
        tensor_y = None
        loaded_metadata = None
        loaded_car_mask = None
        sequences = [torch.randn(64 + i % 3, 6, generator=generator) for i in range(smoke_samples)]
        train_indices, validation_indices = np.arange(len(sequences)), np.asarray([], dtype=int)
    metadata = None
    input_channels = int(sequences[0].shape[-1])
    model_params = getattr(config, "model_params", {}) or {}
    if args.task == "door":
        class_weights = None
        if tensor_y is not None:
            counts = (
                torch.bincount(tensor_y[train_indices].long(), minlength=2).float().clamp_min(1)
            )
            class_weights = counts.sum() / (len(counts) * counts)
        model, loss = (
            DoorNet(input_channels, int(model_params.get("hidden_channels", 16))),
            nn.CrossEntropyLoss(weight=class_weights),
        )
        targets = [
            tensor_y[i].long() if tensor_y is not None else torch.tensor(i % 2)
            for i in range(len(sequences))
        ]
        metadata = [
            loaded_metadata[i].long() if loaded_metadata is not None else torch.tensor(i % 2)
            for i in range(len(sequences))
        ]
    elif args.task == "corrugation":
        class_weights = None
        if tensor_y is not None:
            counts = (
                torch.bincount(tensor_y[train_indices].long(), minlength=3).float().clamp_min(1)
            )
            class_weights = counts.sum() / (len(counts) * counts)
        model, loss = (
            CorrugationNet(input_channels, int(model_params.get("hidden_channels", 16))),
            nn.CrossEntropyLoss(weight=class_weights),
        )
        targets = [
            tensor_y[i].long() if tensor_y is not None else torch.tensor(i % 3)
            for i in range(len(sequences))
        ]
    elif args.task == "shm":
        model, loss = (
            DamageNet(
                input_channels,
                heteroscedastic=False,
                hidden_channels=int(model_params.get("hidden_channels", 16)),
            ),
            smooth_mape,
        )
        targets = [
            tensor_y[i].float() if tensor_y is not None else torch.tensor(float(i) / len(sequences))
            for i in range(len(sequences))
        ]
    else:
        cars, time, channels = tensor_x.shape[1:] if args.preprocessed else (8, 64, 6)
        device = torch.device(selected_device)
        model = ACVRelationalNet(
            channels,
            int(model_params.get("hidden_dim", 16)),
            heads=int(model_params.get("heads", 4)),
        ).to(device)
        x = (
            tensor_x
            if args.preprocessed
            else torch.randn(smoke_samples, cars, time, channels, generator=generator)
        )
        y = tensor_y.long() if tensor_y is not None else torch.arange(smoke_samples) % cars
        x, y = x.to(device), y.to(device)
        if loaded_car_mask is not None:
            car_mask = loaded_car_mask[: args.max_samples] if args.max_samples else loaded_car_mask
            car_mask = car_mask.to(device)
        else:
            car_mask = torch.ones(len(x), cars, dtype=torch.bool, device=device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16
        )
        train_index = torch.as_tensor(train_indices, device=device)
        validation_index = torch.as_tensor(validation_indices, device=device)
        if args.dry_run:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=device.type == "cuda" and amp_dtype != torch.float32,
            ):
                logits = model(x[train_index], car_mask=car_mask[train_index])
                loss = nn.functional.cross_entropy(logits, y[train_index])
            scaler.scale(loss).backward()
            print(
                f"device={device}; amp_dtype={config.runtime.amp_dtype}; dry_run_loss={float(loss.detach()):.6f}"
            )
            return
        best_validation = float("inf")
        history = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=device.type == "cuda" and amp_dtype != torch.float32,
            ):
                logits = model(x[train_index], car_mask=car_mask[train_index])
                loss = nn.functional.cross_entropy(logits, y[train_index])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.eval()
            with torch.no_grad():
                validation_logits = model(x[validation_index], car_mask=car_mask[validation_index])
                validation_loss = nn.functional.cross_entropy(
                    validation_logits, y[validation_index]
                )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss.detach()),
                    "validation_loss": float(validation_loss),
                }
            )
            checkpoint = {
                "model": model.state_dict(),
                "task": "acv",
                "history": history,
                "model_params": model_params,
            }
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, args.output_dir / "last.pt")
            if float(validation_loss) < best_validation:
                best_validation = float(validation_loss)
                torch.save(checkpoint, args.output_dir / "best.pt")
            model.train()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.preprocessed:
            np.savez(
                args.output_dir / "normalization.npz",
                mean=channel_mean.cpu().numpy(),
                std=channel_std.cpu().numpy(),
                train_indices=train_indices,
                validation_indices=validation_indices,
            )
        print(args.output_dir / "best.pt")
        return
    dataset = SequenceDataset(sequences, targets, metadata)
    loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=pad_collate,
    )
    validation_loader = (
        DataLoader(
            Subset(dataset, validation_indices),
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=pad_collate,
        )
        if len(validation_indices)
        else None
    )
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
        np.savez(
            args.output_dir / "normalization.npz",
            mean=channel_mean.cpu().numpy(),
            std=channel_std.cpu().numpy(),
            train_indices=train_indices,
            validation_indices=validation_indices,
        )
    if args.resume:
        trainer.resume(args.resume)
    if args.dry_run:
        print(
            f"device={trainer.device}; amp_dtype={config.runtime.amp_dtype}; dry_run_loss={trainer.dry_run(loader):.6f}"
        )
    else:
        print(trainer.fit(loader, validation_loader))


if __name__ == "__main__":
    main()
