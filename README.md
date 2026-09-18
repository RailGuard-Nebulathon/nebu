# RailGuard AI

RailGuard AI is a reproducible condition-monitoring platform for all four NebulaX 2026 Problem
Statement 3 tracks: Door segment detection/classification, ACV refrigerant-leak localisation,
Rail Corrugation classification, and SHM cumulative-fatigue-damage regression. It prioritises
data correctness, group/file-level validation, strong baselines, official output contracts, and
operator-facing evidence. It never trains automatically and reports no invented scores.

## Architecture

Immutable source files flow through bounded schema inspection and hashed manifests, task adapters,
signal/feature layers, leakage-safe classical or optional compact deep models, calibration/OOD and
evidence layers, then operational inference, the dashboard, and strict submission packaging. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the diagram and module boundaries.

The implementation uses `src/railguard` for reusable package code, `configs` for reproducible YAML,
`scripts` for explicit operations, `dashboard` for the single Streamlit app, `tests` for synthetic
and lightweight official-schema checks, and `outputs` only for generated artifacts.

## Setup

Python 3.11 is the supported target (3.12 is also smoke-tested in this workspace).

```bash
pip install -e ".[all]"
# or a smaller install
pip install -e ".[deep,dev]"
```

Core code works without boosting, Captum, PyWavelets, Optuna, Streamlit, or Plotly. Optional
features fail with actionable installation messages rather than breaking package imports.

## Data placement and inspection

Point `--raw-root` at the supplied `PS3/02_Datasets` folder, whose children are `Door`, `ACV`,
`Rail_Corrugation`, and `SHM`, or mirror that structure under `data/raw`. Raw files remain immutable
and ignored by Git. The official data in this repository were inspected before adapter logic was
written.

```bash
python scripts/inspect_data.py --task all --raw-root <PS3/02_Datasets>
python scripts/build_manifests.py --task all --raw-root <PS3/02_Datasets>
python scripts/build_features.py --task door --raw-root <PS3/02_Datasets> --max-samples 10
```

Reports are written to `data/manifests/schema_report.{json,md}`. ACV mapping discovery writes
`acv_schema_mapping.yaml`. Manifests include SHA-256 file/schema identity and never load all large
files merely to count basic CSV shape.

## Baselines and deep models

Start with fold-safe engineered baselines. The following dry run uses deterministic synthetic data
and does not save or train a competition model:

```bash
python scripts/train_baseline.py --task door --config configs/door/baseline.yaml --dry-run --synthetic
python scripts/train_deep.py --task door --config configs/door/deep.yaml --dry-run
```

Dry-run uses synthetic tensors when no `--preprocessed` NPZ is supplied and says so in the CLI.
Real deep training requires an inspected NPZ with `x`, `y`, and optional masks/metadata; remove
`--dry-run` only when you intentionally want to train. Task strategies are detailed in
[`docs/MODELLING.md`](docs/MODELLING.md). ACV uses within-case candidates and leave-one-case-out
evaluation; corrugation is side-aware and macro-F1-first; SHM uses rainflow proxies without
inventing material constants; Door models operation and normality evidence.

The deep YAML files explicitly request CUDA with A100-friendly BF16 autocast and `high` float32
matmul precision. Use `--device cpu` only as an override. The default
scikit-learn baselines remain CPU models; CUDA acceleration applies to the PyTorch deep/SSL paths.

Build the GPU-training arrays explicitly, for example:

```bash
python scripts/build_sequences.py --task door --raw-root <PS3/02_Datasets> --output data/processed/door_sequences.npz
python scripts/train_deep.py --task door --config configs/door/deep.yaml --preprocessed data/processed/door_sequences.npz
```

The builder uses fixed-length interpolation without modifying source files. Training persists the
per-channel normalization statistics next to its checkpoints.

Self-supervised learning and hyperparameter optimisation are optional, explicit commands:

```bash
python scripts/pretrain_ssl.py --task corrugation --config configs/corrugation/dual_domain.yaml --dry-run
```

## Evaluation, robustness, and evidence

Splits are made at segment/file/case/group level. Preprocessors are contained inside pipelines,
duplicate hashes/groups are rejected, and calibration requires held-out predictions.

```bash
python scripts/evaluate.py --task shm --truth truth.csv --predictions predictions.csv
python scripts/robustness.py --task door --checkpoint <bundle>
python scripts/explain.py --task door --checkpoint <bundle>
python scripts/calibrate.py --calibration held_out_residuals.csv
```

See [`docs/EVALUATION.md`](docs/EVALUATION.md). The platform includes official Door IoU-weighted
F1, ACV rank decay, corrugation macro F1, SHM `max(0, 1-MAPE)`, bootstrap intervals, calibration,
ensemble uncertainty, two OOD scores, corruption retention, and non-causal model evidence.

## Inference

Bundles are transparent directories containing metadata, config, schema, feature names,
preprocessing, and model files. Load only trusted local joblib/PyTorch bundles.

```python
from railguard.inference import RailGuardPredictor

predictor = RailGuardPredictor.from_bundle("outputs/checkpoints/my_bundle")
result = predictor.predict(task="door", input_path="data/raw/Door/Test.csv")
```

CLI equivalents are `python scripts/infer.py ...` and the hackathon-compatible root `predict.py`.
Inference rejects a task or feature schema incompatible with the bundle.

## Dashboard and compulsory app workflow

```bash
streamlit run dashboard/app.py
```

Demo mode is checkpoint-free and labels all values as simulation. Real mode reads generated
outputs or accepts a source-file upload plus a trusted bundle, renders results, and provides a
download. Views cover fleet, Door, ACV, corrugation, SHM, reliability, and transparent health.
Record the required ≤3-minute demo video externally; a source repository cannot fabricate that
human-recorded deliverable. See [`docs/DASHBOARD.md`](docs/DASHBOARD.md).

## Submission

The official schemas and filenames come from the supplied Info Kits/examples—not guesses:
`door_predictions.csv`, `acv_predictions.csv`, `rail_predictions.csv`, and `shm_predictions.csv`.

```bash
python scripts/generate_submission.py --task all --predictions-root outputs/predictions
python scripts/validate_submission.py outputs/submissions/predictions.zip
```

The ZIP is flat, IDs are complete and unique, row order is template-preserving, ACV car IDs retain
leading zeros, and labels/types are strict. See [`docs/SUBMISSION.md`](docs/SUBMISSION.md).

## Testing

```bash
pytest -q -p no:cacheprovider
ruff check .
python -m compileall -q src dashboard scripts
```

Tests do not launch expensive training. The real-data checks read only a few representative
official files; the end-to-end test is synthetic and CPU-small.

## Known limitations

- No leaderboard performance is claimed; full training/tuning remains a deliberate user action.
- ACV has only six labelled cases, so the relational baseline is statistically preferable to a
  high-capacity network until more cases exist.
- Deep models and SSL are architecture/smoke-tested, not asserted superior to baselines.
- Streamlit and Ruff are optional dependencies and must be installed to launch/lint respectively.
- A trained bundle, final held-out inference, and the human-recorded demo video cannot be produced
  without the user explicitly training/selecting a model and recording the app.
