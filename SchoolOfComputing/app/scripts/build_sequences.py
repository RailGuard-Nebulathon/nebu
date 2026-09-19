"""Build fixed-length, task-aware NPZ arrays for explicit deep training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from railguard.data.adapters.acv import ACVAdapter, CAR_COLUMN, split_car_columns
from railguard.data.adapters.corrugation import CorrugationAdapter
from railguard.data.adapters.door import DoorAdapter
from railguard.data.adapters.shm import SHMAdapter
from railguard.features.acv import ACV_SEQUENCE_SCHEMA, ACVSequenceCanonicalizer
from railguard.signal.resampling import resample_length


def _limit(values: list, maximum: int | None) -> list:
    return values[:maximum] if maximum else values


def _door(root: Path, length: int, maximum: int | None) -> dict[str, np.ndarray]:
    samples = _limit(list(DoorAdapter(root / "Door").samples("train")), maximum)
    return {
        "x": np.stack([resample_length(sample.values, length) for sample in samples]).astype(np.float32),
        "y": np.asarray([0 if sample.target == "Normal" else 1 for sample in samples], dtype=np.int64),
        "metadata": np.asarray([0 if sample.metadata["operation"] == "Open" else 1 for sample in samples], dtype=np.int64),
        "sample_ids": np.asarray([sample.sample_id for sample in samples]),
        "channel_names": np.asarray(samples[0].channel_names),
    }


def _corrugation(root: Path, length: int, maximum: int | None) -> dict[str, np.ndarray]:
    samples = []
    for sample in CorrugationAdapter(root / "Rail_Corrugation").samples("train"):
        samples.append(sample)
        if maximum and len(samples) >= maximum:
            break
    labels = {"Normal": 0, "Side I": 1, "Side II": 2}
    return {
        "x": np.stack([resample_length(sample.values, length) for sample in samples]).astype(np.float32),
        "y": np.asarray([labels[str(sample.target)] for sample in samples], dtype=np.int64),
        "sample_ids": np.asarray([sample.sample_id for sample in samples]),
        "channel_names": np.asarray(samples[0].channel_names),
    }


def _shm(root: Path, length: int, maximum: int | None) -> dict[str, np.ndarray]:
    samples = []
    for sample in SHMAdapter(root / "SHM").samples("train"):
        samples.append(sample)
        if maximum and len(samples) >= maximum:
            break
    return {
        "x": np.stack([resample_length(sample.values, length) for sample in samples]).astype(np.float32),
        "y": np.asarray([sample.target for sample in samples], dtype=np.float32),
        "sample_ids": np.asarray([sample.sample_id for sample in samples]),
        "channel_names": np.asarray(samples[0].channel_names),
    }


def _acv(root: Path, length: int, maximum: int | None) -> dict[str, np.ndarray]:
    adapter = ACVAdapter(root / "ACV")
    files = _limit(adapter.files("train"), maximum)
    labels = pd.read_csv(adapter.root / "Train_Labels.csv", dtype={"faulty_car": str}).set_index("filename")["faulty_car"].str.zfill(2)
    cases = []
    for path in files:
        aliases = {alias for _, names in ACV_SEQUENCE_SCHEMA.values() for alias in names}
        header = pd.read_excel(path, nrows=0)
        selected = [
            column for column in header.columns.astype(str)
            if (match := CAR_COLUMN.match(column)) and match.group("parameter") in aliases
        ]
        if not selected:
            raise ValueError(f"No canonical ACV sequence columns were found in {path.name}")
        _, cars = split_car_columns(pd.read_excel(path, usecols=selected))
        cases.append((path, cars))
    canonicalizer = ACVSequenceCanonicalizer().fit(
        frame for _, cars in cases for frame in cars.values()
    )
    arrays, targets, car_ids = [], [], []
    source_schema: dict[str, dict[str, dict[str, str | None]]] = {}
    inactive_cars: dict[str, list[str]] = {}
    for path, cars in cases:
        identifiers = []
        per_car = []
        source_schema[path.name] = {}
        inactive_cars[path.name] = []
        for identifier in sorted(cars):
            canonical, sources = canonicalizer.transform(cars[identifier])
            source_schema[path.name][identifier] = sources
            signal_count = len(canonicalizer.channel_names) // 2
            if not canonical[:, signal_count:].any():
                inactive_cars[path.name].append(identifier)
                continue
            identifiers.append(identifier)
            per_car.append(resample_length(canonical, length))
        faulty_car = labels[path.name]
        if faulty_car not in identifiers:
            raise ValueError(f"Labelled faulty car {faulty_car} has no ACV data in {path.name}")
        arrays.append(np.stack(per_car)); targets.append(identifiers.index(labels[path.name])); car_ids.append(identifiers)
    max_cars = max(len(case) for case in arrays)
    padded = np.zeros((len(arrays), max_cars, length, len(canonicalizer.channel_names)), dtype=np.float32)
    car_mask = np.zeros((len(arrays), max_cars), dtype=bool)
    padded_ids = np.full((len(arrays), max_cars), "", dtype=f"<U{max(len(car) for ids in car_ids for car in ids)}")
    for index, (case, identifiers) in enumerate(zip(arrays, car_ids, strict=True)):
        padded[index, : len(case)] = case
        car_mask[index, : len(case)] = True
        padded_ids[index, : len(identifiers)] = identifiers
    return {
        "x": padded, "y": np.asarray(targets, dtype=np.int64), "car_mask": car_mask,
        "sample_ids": np.asarray([path.name for path, _ in cases]), "car_ids": padded_ids,
        "channel_names": np.asarray(canonicalizer.channel_names),
        "category_maps_json": np.asarray(json.dumps(canonicalizer.category_maps, sort_keys=True)),
        "source_schema_json": np.asarray(json.dumps(source_schema, sort_keys=True)),
        "inactive_cars_json": np.asarray(json.dumps(inactive_cars, sort_keys=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("door", "acv", "corrugation", "shm"))
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int)
    parser.add_argument("--max-samples", type=int, help="Lightweight inspection/smoke subset")
    args = parser.parse_args()
    default_length = {"door": 192, "acv": 256, "corrugation": 1024, "shm": 2048}[args.task]
    builder = {"door": _door, "acv": _acv, "corrugation": _corrugation, "shm": _shm}[args.task]
    arrays = builder(args.raw_root, args.length or default_length, args.max_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"{args.task}: x={arrays['x'].shape}, y={arrays['y'].shape} -> {args.output}")


if __name__ == "__main__":
    main()
