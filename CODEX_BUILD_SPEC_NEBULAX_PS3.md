# NebulaX 2026 — Problem Statement 3
# Codex Master Build Specification

> **Purpose of this file:** Give this entire document to Codex as the implementation contract for the project.
>
> **Primary goal:** Build a technically advanced, competition-grade, reproducible train-condition-monitoring system for NebulaX 2026 Problem Statement 3.
>
> **Important:** Do **not** automatically launch expensive model training. Build the full project, data pipeline, models, CLIs, evaluation framework, dashboard, tests, and documentation so that training can be started explicitly by the user later.

---

## 0. Codex operating instructions

You are the principal software engineer implementing this repository.

Work through this specification systematically. Do not replace the requested architecture with a toy notebook or a single monolithic script.

### Non-negotiable rules

1. **Do not fabricate data schemas, labels, metrics, or model results.**
2. **Do not auto-run expensive training.**
3. Lightweight unit tests, import checks, tiny synthetic smoke tests, and schema-inspection commands are allowed.
4. If real raw data are present, inspect their schema before writing subsystem-specific parsing logic.
5. If real raw data are not present, implement adapters that fail with clear messages and include synthetic fixtures for tests.
6. Keep all raw data out of Git.
7. Never modify raw source files in place.
8. Every transformation must be reproducible from configuration.
9. Prevent target leakage and train/test contamination.
10. Use entity/group-aware or chronological validation where the data support it; do not casually random-split time-series rows.
11. Every model must expose a common inference interface.
12. Every output submission must be validated against an explicit schema before packaging.
13. Do not report fake leaderboard performance in README files.
14. Keep advanced features modular. A missing optional dependency must not make the whole repository unusable.
15. Prefer clean, typed, tested Python modules over notebooks.
16. Use deterministic seeds wherever practical.
17. Log assumptions and discovered schemas to machine-readable files.
18. Preserve the official filenames/IDs needed for submission.
19. Build all four problem tracks:
    - Door anomaly detection
    - ACV refrigerant-leak localisation
    - Rail corrugation classification
    - Structural health monitoring fatigue-damage regression
20. Treat the four tracks as one shared predictive-maintenance platform, while keeping their training and submission logic independent.

### How to proceed

Implement the repository in the phased order defined near the end of this document. After each phase:

- run relevant unit tests;
- run lint/type/import checks where practical;
- update `docs/IMPLEMENTATION_STATUS.md`;
- do not start the next phase if the current phase leaves broken imports or tests.

If a requirement is impossible without seeing the raw data, implement:
- a robust interface,
- schema introspection,
- validation,
- clear TODO generated from inspection,
- synthetic tests,

rather than inventing the missing schema.

---

# 1. Challenge summary

Build a predictive-maintenance / train-condition-monitoring platform for four independent rail-vehicle subsystems.

The competition tasks are:

| Track | Task | Primary ML formulation |
|---|---|---|
| Door | Detect abnormal-resistance door open/close cycles | Segmentation + binary classification |
| ACV | Identify the car containing a refrigerant leak | Fault localisation / multiclass classification |
| Rail Corrugation | Classify Normal vs Side I vs Side II corrugation | Imbalanced multiclass time-series classification |
| SHM | Estimate cumulative fatigue damage | Time-series regression |

The competition outputs are the first priority. The wider system should additionally provide:

- confidence / uncertainty;
- explainability;
- robustness diagnostics;
- degradation / health scores;
- model comparison;
- operator-friendly visualisation;
- reproducible experiment tracking;
- submission validation.

The project should feel like an industrial predictive-maintenance platform, not four unrelated notebooks.

---
# 2. Data
All relevant data is found under PS3


---

# 3. Product vision

Working title:

```text
RailGuard AI
```

The architecture should support a polished story:

> Raw train telemetry is converted into event-level diagnoses, fault localisation, degradation estimates, calibrated confidence, root-cause evidence, and fleet-level maintenance intelligence.

High-level flow:

```text
Raw railway sensor data
        │
        ▼
Schema inspection + validation
        │
        ▼
Signal conditioning / cycle or window construction
        │
        ├───────────────┬──────────────────┬──────────────────┐
        ▼               ▼                  ▼                  ▼
      Door             ACV            Corrugation            SHM
        │               │                  │                  │
        ▼               ▼                  ▼                  ▼
task-specific feature extraction + task-specific model heads
        │
        └───────────────┴──────────────────┴──────────────────┘
                                │
                                ▼
                  Calibration / uncertainty / OOD
                                │
                                ▼
                     Explainability / evidence
                                │
                                ▼
                      Unified health intelligence
                                │
                                ▼
                 Operator dashboard + submissions
```

---

# 4. Technical priorities

The repository should demonstrate strength in the following areas.

## 4.1 Strong baselines before exotic models

Implement:
- statistical features;
- temporal features;
- spectral features;
- wavelet features;
- gradient / derivative features;
- robust tree-based models;
- linear baselines where useful.

Optional model libraries should support:
- scikit-learn;
- XGBoost if installed;
- LightGBM if installed;
- CatBoost if installed.

There must always be a scikit-learn fallback.

## 4.2 Multi-resolution temporal modelling

Implement reusable deep-learning components:
- 1D convolution stem;
- residual temporal blocks;
- dilated TCN blocks;
- Transformer encoder option;
- mask-aware pooling;
- multi-scale feature extraction.

## 4.3 Time-frequency modelling

Implement:
- FFT power spectra;
- Welch PSD;
- STFT;
- wavelet feature extraction;
- frequency band aggregation;
- spectral entropy;
- dominant frequencies;
- optional learned spectral branch.

## 4.4 Sensor/channel reasoning

Implement:
- per-channel normalisation;
- channel masking;
- channel-attention block;
- missing-channel robustness;
- sensor attribution.

## 4.5 Uncertainty and out-of-distribution awareness

Implement:
- probability calibration for classification;
- ensemble or conformal-style uncertainty hooks;
- regression prediction intervals;
- embedding-distance OOD scoring;
- abstention thresholds.

## 4.6 Explainability

Implement:
- feature importance for tree models;
- permutation importance;
- Integrated Gradients if Captum is installed;
- temporal saliency;
- channel importance;
- nearest-neighbour retrieval in embedding space;
- spectral-band attribution.

## 4.7 Robustness

Evaluation should optionally inject:
- Gaussian noise;
- amplitude scaling;
- sensor bias drift;
- random point dropout;
- contiguous missing intervals;
- timestamp jitter;
- downsampling;
- single-channel dropout.

## 4.8 Reproducibility

Implement:
- YAML configuration;
- seeds;
- saved preprocessors;
- model metadata;
- environment version capture;
- experiment IDs;
- manifest hashes;
- git commit hash when available.

---

# 5. Repository layout

Create this structure. Minor additions are allowed, but do not collapse it into a smaller architecture.

```text
nebulathon_ps3/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── Makefile
│
├── configs/
│   ├── base.yaml
│   ├── paths.example.yaml
│   ├── door/
│   │   ├── baseline.yaml
│   │   ├── deep.yaml
│   │   └── ensemble.yaml
│   ├── acv/
│   │   ├── baseline.yaml
│   │   └── relational.yaml
│   ├── corrugation/
│   │   ├── baseline.yaml
│   │   ├── dual_domain.yaml
│   │   └── ensemble.yaml
│   └── shm/
│       ├── baseline.yaml
│       ├── deep.yaml
│       └── ensemble.yaml
│
├── data/
│   ├── raw/
│   │   ├── door/
│   │   ├── acv/
│   │   ├── corrugation/
│   │   └── shm/
│   ├── interim/
│   ├── processed/
│   ├── manifests/
│   └── README.md
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA_CONTRACTS.md
│   ├── MODELLING.md
│   ├── EVALUATION.md
│   ├── SUBMISSION.md
│   ├── DASHBOARD.md
│   ├── IMPLEMENTATION_STATUS.md
│   └── ASSUMPTIONS.md
│
├── src/
│   └── railguard/
│       ├── __init__.py
│       ├── constants.py
│       ├── types.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── schema.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── registry.py
│       │   ├── manifest.py
│       │   ├── schema_inspector.py
│       │   ├── validation.py
│       │   ├── timestamp.py
│       │   ├── splits.py
│       │   ├── scaling.py
│       │   ├── sequence.py
│       │   ├── synthetic.py
│       │   └── adapters/
│       │       ├── __init__.py
│       │       ├── base.py
│       │       ├── door.py
│       │       ├── acv.py
│       │       ├── corrugation.py
│       │       └── shm.py
│       │
│       ├── signal/
│       │   ├── __init__.py
│       │   ├── cleaning.py
│       │   ├── filtering.py
│       │   ├── resampling.py
│       │   ├── derivatives.py
│       │   ├── fft.py
│       │   ├── stft.py
│       │   ├── wavelets.py
│       │   ├── spectral.py
│       │   └── cycle_normalization.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── statistical.py
│       │   ├── temporal.py
│       │   ├── spectral.py
│       │   ├── wavelet.py
│       │   ├── cross_channel.py
│       │   ├── door.py
│       │   ├── acv.py
│       │   ├── corrugation.py
│       │   └── shm.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── protocol.py
│       │   ├── factory.py
│       │   ├── serialization.py
│       │   ├── classical/
│       │   │   ├── __init__.py
│       │   │   ├── classifier.py
│       │   │   ├── regressor.py
│       │   │   └── anomaly.py
│       │   ├── backbones/
│       │   │   ├── __init__.py
│       │   │   ├── cnn1d.py
│       │   │   ├── tcn.py
│       │   │   ├── transformer.py
│       │   │   ├── spectral_encoder.py
│       │   │   ├── channel_attention.py
│       │   │   ├── fusion.py
│       │   │   └── pooling.py
│       │   ├── door/
│       │   │   ├── __init__.py
│       │   │   ├── normality.py
│       │   │   ├── detector.py
│       │   │   └── ensemble.py
│       │   ├── acv/
│       │   │   ├── __init__.py
│       │   │   ├── relational.py
│       │   │   ├── graph.py
│       │   │   └── localizer.py
│       │   ├── corrugation/
│       │   │   ├── __init__.py
│       │   │   ├── dual_domain.py
│       │   │   └── classifier.py
│       │   └── shm/
│       │       ├── __init__.py
│       │       ├── physics_features.py
│       │       ├── damage_regressor.py
│       │       └── ensemble.py
│       │
│       ├── ssl/
│       │   ├── __init__.py
│       │   ├── augmentations.py
│       │   ├── masked_reconstruction.py
│       │   ├── contrastive.py
│       │   └── pretrainer.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── seed.py
│       │   ├── datasets.py
│       │   ├── losses.py
│       │   ├── trainer.py
│       │   ├── callbacks.py
│       │   ├── early_stopping.py
│       │   └── optimization.py
│       │
│       ├── calibration/
│       │   ├── __init__.py
│       │   ├── temperature.py
│       │   ├── isotonic.py
│       │   ├── conformal.py
│       │   └── metrics.py
│       │
│       ├── uncertainty/
│       │   ├── __init__.py
│       │   ├── ensembles.py
│       │   ├── mc_dropout.py
│       │   ├── regression.py
│       │   └── ood.py
│       │
│       ├── explainability/
│       │   ├── __init__.py
│       │   ├── feature_importance.py
│       │   ├── temporal_saliency.py
│       │   ├── integrated_gradients.py
│       │   ├── spectral_attribution.py
│       │   └── neighbours.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── classification.py
│       │   ├── regression.py
│       │   ├── bootstrap.py
│       │   ├── robustness.py
│       │   ├── calibration.py
│       │   ├── error_analysis.py
│       │   ├── ablation.py
│       │   └── report.py
│       │
│       ├── health/
│       │   ├── __init__.py
│       │   ├── scoring.py
│       │   ├── trend.py
│       │   └── maintenance.py
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── predictor.py
│       │   ├── batch.py
│       │   └── schemas.py
│       │
│       ├── submission/
│       │   ├── __init__.py
│       │   ├── schemas.py
│       │   ├── writer.py
│       │   ├── validator.py
│       │   └── packager.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           ├── hashing.py
│           ├── io.py
│           ├── environment.py
│           └── optional.py
│
├── scripts/
│   ├── inspect_data.py
│   ├── build_manifests.py
│   ├── build_features.py
│   ├── train_baseline.py
│   ├── train_deep.py
│   ├── pretrain_ssl.py
│   ├── calibrate.py
│   ├── evaluate.py
│   ├── robustness.py
│   ├── explain.py
│   ├── infer.py
│   ├── generate_submission.py
│   └── validate_submission.py
│
├── dashboard/
│   ├── app.py
│   ├── state.py
│   ├── data_access.py
│   ├── components/
│   │   ├── fleet_overview.py
│   │   ├── sensor_view.py
│   │   ├── anomaly_view.py
│   │   ├── explanation_view.py
│   │   └── health_view.py
│   └── README.md
│
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_timestamp.py
│   ├── test_manifest.py
│   ├── test_door_adapter.py
│   ├── test_acv_adapter.py
│   ├── test_corrugation_adapter.py
│   ├── test_shm_adapter.py
│   ├── test_features.py
│   ├── test_models_smoke.py
│   ├── test_metrics.py
│   ├── test_submission.py
│   └── test_end_to_end_synthetic.py
│
├── outputs/
│   ├── checkpoints/
│   ├── experiments/
│   ├── figures/
│   ├── predictions/
│   ├── reports/
│   └── submissions/
│
└── notebooks/
    └── README.md
```

---

# 6. Packaging and dependencies

Use Python 3.11 unless a required dependency forces otherwise.

Use `pyproject.toml`.

Core dependencies:

```text
numpy
pandas
scipy
scikit-learn
pyyaml
pydantic
joblib
matplotlib
tqdm
openpyxl
pyarrow
```

Deep-learning extra:

```text
torch
```

Optional extras:

```text
xgboost
lightgbm
catboost
pywavelets
captum
optuna
streamlit
plotly
networkx
```

Design optional imports gracefully.

Recommended extras groups:

```toml
[project.optional-dependencies]
deep = [...]
boosting = [...]
explain = [...]
dashboard = [...]
dev = [...]
all = [...]
```

Developer dependencies:

```text
pytest
pytest-cov
ruff
mypy
pre-commit
```

Do not require MLflow or Weights & Biases. Local JSON/CSV experiment logging is mandatory; external tracking can be optional.

---

# 7. Configuration system

Use YAML files validated by Pydantic/dataclasses.

Base config must cover:

```yaml
project:
  name: railguard
  seed: 42

paths:
  data_root: data
  raw_root: data/raw
  processed_root: data/processed
  manifest_root: data/manifests
  output_root: outputs

runtime:
  device: auto
  num_workers: 0
  deterministic: true

logging:
  level: INFO

training:
  epochs: 100
  batch_size: 32
  early_stopping_patience: 12
  learning_rate: 0.001
  weight_decay: 0.0001

evaluation:
  bootstrap_samples: 1000
  calibration_bins: 10

robustness:
  enabled: false
```

Subsystem configs override relevant fields.

Implement config precedence:

```text
defaults < YAML < CLI overrides
```

Resolved configuration must be saved with each experiment.

---

# 8. Data contracts

## 8.1 Canonical sample types

Create typed structures in `types.py`.

Examples:

```python
@dataclass
class SequenceSample:
    sample_id: str
    values: np.ndarray       # [T, C]
    timestamps: np.ndarray | None
    channel_names: list[str]
    metadata: dict[str, Any]
    target: Any | None = None
```

```python
@dataclass
class Prediction:
    sample_id: str
    task: str
    prediction: Any
    probabilities: dict[str, float] | None
    confidence: float | None
    uncertainty: float | None
    ood_score: float | None
    metadata: dict[str, Any]
```

Do not force every subsystem into identical raw shapes; unify them at the model/inference boundary.

## 8.2 Dataset manifest

Every raw-file collection should be represented by a manifest with fields such as:

```text
sample_id
task
source_path
split
label
group_id
n_rows
n_channels
schema_hash
file_hash
metadata_json
```

For regression, `label` may hold numeric target or use `target`.

Use SHA-256 or a fast cryptographic hash for files/schemas where practical.

## 8.3 Schema inspector

`schema_inspector.py` must inspect:

- extension;
- sheets for Excel;
- columns;
- dtypes;
- row counts;
- null percentages;
- constant columns;
- candidate timestamp columns;
- candidate target leakage columns;
- numeric/non-numeric channels;
- sampling interval statistics;
- duplicated timestamps;
- monotonicity;
- basic min/max/mean/std.

Write results to:

```text
data/manifests/schema_report.json
data/manifests/schema_report.md
```

The inspection script must work without loading all huge files into RAM if files are large.

---

# 9. Timestamp handling

The door timestamp format is non-standard.

Implement a robust parser for strings shaped like:

```text
YYYY-M-D-H-M-S-ms
```

Example:

```text
2023-7-5-0-0-0-20
```

Parse from the right or with a validated regex, not naive generic `pd.to_datetime`.

Support:
- variable digit widths;
- millisecond suffix;
- malformed-row diagnostics;
- conversion to elapsed seconds;
- monotonicity checks.

For other tasks, infer time columns but require config if ambiguous.

---

# 10. Door pipeline

## 10.1 Objective

Detect abnormal-resistance door operations and produce segment-level classification.

Known classes:

```text
Normal
Abnormal resistance
```

Known operation types:

```text
Open
Close
```

## 10.2 Segment extraction

Build a `DoorAdapter`.

Training:

1. Read `Train.csv`.
2. Parse timestamps.
3. Read `Train_Segments_Answer.csv`.
4. Slice the raw time-series by `start_time` and `end_time`.
5. Validate sliced row counts against `n_rows`.
6. Log any off-by-one/time-boundary mismatch.
7. Store each segment as a `SequenceSample`.

Test:

- If official test segment definitions exist, use them.
- If not, infer cycles from:
  - Open/Close command;
  - Door is opening / closing;
  - position transitions;
  - terminal state switches.
- Make segmentation configurable and inspectable.

Do not use status labels in segmentation logic.

## 10.3 Door channel groups

Default continuous channels:

```text
Motor current(mA)
Motor Voltage(10mV)
Motor electrodynamic force
Door opening time(.1s)
Door closing time(.1s)
Door leaf position
```

Default binary/state channels:

```text
Close command
Open command
DCSR
DCSL
DLSR
DLSL
Door Opened
Door Locked
Door is opening
Door is closing
```

Allow overrides from YAML.

## 10.4 Phase-normalised representation

Open and close cycles have different lengths.

Implement:
- raw variable-length representation + mask;
- optional interpolation to canonical length;
- operation-aware phase coordinate `0..1`;
- derivatives of position/current/voltage/EMF.

Canonical length should be configurable, e.g. 192, not hard-coded into algorithms.

Never mix Open and Close without giving model the operation type or training operation-specific models.

## 10.5 Door engineered features

At minimum compute:

### Current / voltage / EMF
- mean;
- std;
- median;
- MAD;
- min/max;
- range;
- RMS;
- energy;
- integral / area under curve;
- positive area;
- negative area;
- peak magnitude;
- peak timing as fraction of cycle;
- quantiles;
- skewness;
- kurtosis;
- crest factor;
- impulse factor;
- shape factor;
- derivative mean/std/max;
- count of local peaks.

### Position
- total travel;
- monotonicity violations;
- average velocity;
- max velocity;
- average acceleration;
- max acceleration;
- stall-like low-position-change intervals;
- current-vs-position relation.

### Physics-inspired
- approximate electrical power proxy: current × voltage;
- energy proxy over the cycle;
- current per unit position change;
- work/resistance proxies;
- cross-correlation current vs position derivative;
- lag of maximum correlation.

Treat units carefully: do not claim true mechanical work unless the necessary physical parameters exist.

### Spectral
- dominant frequency;
- spectral centroid;
- spectral bandwidth;
- spectral entropy;
- band powers;
- high-frequency power ratio.

## 10.6 Door baseline models

Implement:
- logistic regression;
- random forest / ExtraTrees;
- HistGradientBoosting;
- optional XGBoost/LightGBM/CatBoost.

Use class weights where appropriate.

Validation:
- stratify by status;
- preserve operation distribution;
- if later group IDs exist, group by train/door/recording source.

Metrics:
- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1;
- AUROC;
- AUPRC;
- confusion matrix;
- per-operation metrics.

AUPRC and recall for abnormal resistance should be prominent because the positive class is the fault.

## 10.7 Deep DoorNet

Implement a modular model:

```text
continuous signals
    ↓
multi-scale CNN / TCN encoder
    ↓
mask-aware temporal pooling
    ┐
    ├─ fusion ─ classifier
spectral branch
    ┘
```

Add:
- operation embedding;
- optional state-channel encoder;
- channel attention.

Suggested blocks:
- kernels 3, 7, 15 in parallel OR stacked dilated convolutions;
- residual connections;
- LayerNorm/BatchNorm where appropriate;
- dropout;
- attention pooling.

Do not over-size the network given only 110 labelled segments.

## 10.8 Door normality model

Implement an optional one-class/normality path trained on Normal cycles:
- PCA reconstruction;
- IsolationForest;
- autoencoder if enough data;
- distance from normal embedding centroid.

Create a combined anomaly score:

```text
final_score =
    w_supervised * classifier_probability
  + w_normality * normalized_anomaly_score
```

Weights are fit only on training/validation data.

## 10.9 Door explanation

Provide:
- feature importance;
- current/position overlay;
- abnormal saliency by phase;
- top anomalous phase window;
- nearest normal segment;
- nearest abnormal segment if available.

---

# 11. ACV pipeline

## 11.1 Objective

Given one ACV case, identify which car has a refrigerant leak.

Known label:

```text
faulty_car: integer car identifier
```

Known labelled cases: 6 in supplied label file.

## 11.2 Mandatory schema discovery

ACV data are `.xlsx`.

The adapter must inspect:
- sheet names;
- row/column dimensions;
- time columns;
- car identifiers embedded in column names or sheets;
- cabin temperatures;
- ambient temperature;
- control mode;
- setpoints;
- compressor/control signals;
- any status fields.

Generate a discovered mapping file:

```text
data/manifests/acv_schema_mapping.yaml
```

If cars are represented as:
- separate sheets,
- prefixed columns,
- long-form rows,

support configuration for all three patterns.

## 11.3 Relative/relational modelling

With only a handful of labelled cases, default features should compare each car to peer cars in the same case.

For car `i`, generate:

```text
car absolute features
minus
fleet/case peer median features
```

Useful feature families:
- cabin temperature mean/min/max/std;
- temperature slope;
- time above setpoint;
- recovery time after control transition;
- oscillation amplitude;
- lagged response;
- cabin minus ambient temperature;
- cabin minus setpoint;
- car minus peer median;
- car minus nearest healthy cluster;
- control-state-conditioned residuals;
- autocorrelation;
- change-point features.

Create one candidate row per `(case, car)`.

Training labels:

```text
is_faulty = (car_id == faulty_car)
```

Then rank all cars within a case.

This turns ACV into a **grouped learning-to-rank / candidate classification** problem rather than a brittle direct six-class classifier.

## 11.4 ACV baseline

Implement:
- robust z-score / rule score;
- logistic regression;
- random forest / ExtraTrees;
- gradient boosting.

Evaluation must be leave-one-case-out if dataset size remains tiny.

Metrics:
- top-1 localisation accuracy;
- top-2 localisation accuracy;
- mean reciprocal rank;
- rank of true faulty car;
- confidence margin between best and second candidate.

## 11.5 Relational ACV neural model

Build but do not make it the default with six labelled cases.

Architecture:

```text
per-car temporal encoder
        ↓
car embeddings
        ↓
cross-car self-attention
        ↓
per-car fault logits
        ↓
softmax across cars in the case
```

Mask variable number of cars.

Optionally support a graph:
- each car = node;
- edges connect neighbouring or all cars;
- node features = temporal embedding + engineered residual features;
- graph attention or simple message passing.

Implement without requiring PyTorch Geometric; use native PyTorch so setup remains simple.

## 11.6 ACV explanation

Show:
- ranking of cars;
- key relative features;
- cabin temperature traces;
- peer median trace;
- residual trace;
- control-state periods;
- confidence gap.

---

# 12. Rail corrugation pipeline

## 12.1 Objective

Classify each labelled sensor file as:

```text
Normal
Side I
Side II
```

Known imbalance:

```text
Normal: 234
Side II: 24
Side I: 14
```

## 12.2 Data adapter

Read each file referenced by the label CSV.

Inspect:
- time column;
- sampling frequency;
- vibration / acceleration / shock channels;
- axis identifiers;
- side identifiers;
- constant/non-sensor columns.

If sampling frequency is not explicit:
- infer from timestamp deltas;
- otherwise require YAML setting.

## 12.3 Signal conditioning

Configurable:
- detrending;
- de-meaning;
- robust scaling;
- optional high-pass / band-pass;
- clipping/winsorisation;
- resampling;
- windowing with overlap.

Never filter with parameters that require unknown physical assumptions without making them configurable.

## 12.4 Engineered features

Per channel:
- mean/std/RMS;
- peak-to-peak;
- skew;
- kurtosis;
- crest factor;
- impulse factor;
- clearance factor;
- shape factor;
- zero crossing rate;
- autocorrelation peaks;
- entropy.

Frequency:
- PSD;
- dominant peaks;
- spectral centroid;
- spectral entropy;
- spectral rolloff;
- configurable band powers;
- ratios between low/mid/high-frequency energy.

Time-frequency:
- STFT summary;
- wavelet packet energies if PyWavelets installed.

Cross-channel:
- correlation matrix summary;
- coherence if practical;
- side/axis energy ratios.

## 12.5 Baseline classifier

Implement:
- class-weighted logistic regression;
- ExtraTrees;
- HistGradientBoosting;
- optional CatBoost/LightGBM/XGBoost.

Validation:
- StratifiedKFold only if filenames represent truly independent units.
- If metadata reveal common runs/trains/sections, switch to StratifiedGroupKFold or GroupKFold.
- Log split reasoning.

Metrics:
- macro F1 **primary**;
- balanced accuracy;
- per-class precision/recall/F1;
- confusion matrix;
- one-vs-rest AUROC where mathematically valid;
- AUPRC per class;
- normal-vs-abnormal secondary metrics.

Do not rely on raw accuracy because Normal dominates.

## 12.6 Dual-domain CorrugationNet

Architecture:

```text
raw multichannel sequence ── temporal CNN/TCN ──┐
                                                ├─ gated attention fusion ─ classifier
STFT/PSD representation ── spectral encoder ────┘
```

Components:
- multiscale temporal kernels;
- channel attention;
- frequency encoder;
- learned fusion gate;
- class-weighted cross entropy or focal loss;
- label smoothing configurable.

Use moderate capacity.

Support window-level logits aggregated to file-level by:
- probability mean;
- logit mean;
- attention pooling.

The official prediction must be file-level.

## 12.7 Imbalance handling

Config options:
- class weights;
- weighted sampler;
- focal loss;
- balanced batches.

Do not perform naive oversampling before validation split.

## 12.8 Corrugation explanation

Generate:
- most influential sensor channels;
- frequency bands;
- file-level spectral plot;
- temporal saliency;
- confusion examples;
- nearest training embeddings.

---

# 13. Structural Health Monitoring pipeline

## 13.1 Objective

Predict continuous cumulative fatigue damage for each file.

Known target:

```text
damage: float
```

Known labelled files: 64.

## 13.2 Adapter and schema inspection

Identify:
- time;
- stress/strain channels;
- load channels;
- relevant sensor IDs;
- sampling frequency.

Do not assume a single stress column.

Allow config:

```yaml
shm:
  time_column: null
  stress_columns: auto
  strain_columns: auto
```

`auto` means schema discovery with a clear report.

## 13.3 Physics-informed feature layer

If stress-like signals are available, implement:

- mean;
- RMS;
- range;
- max absolute stress;
- standard deviation;
- quantiles;
- positive/negative excursion counts;
- derivative statistics;
- stress-range histogram;
- cycle-count features.

Implement rainflow counting internally or behind an optional lightweight dependency.

If S-N curve parameters are **not** given:
- do not invent material constants;
- derive generic cycle-amplitude distribution features;
- call them fatigue proxies.

If S-N parameters are later configured:
- support Miner-style cumulative damage estimate;
- expose it as an engineered feature;
- optionally use it in a physics-consistency loss.

## 13.4 Baseline regression

Implement:
- Ridge/ElasticNet;
- RandomForestRegressor / ExtraTreesRegressor;
- HistGradientBoostingRegressor;
- optional XGBoost/LightGBM/CatBoost.

Because the target is skewed, config may support:
- raw target;
- log1p target if valid;
- quantile transformation.

Any inverse transform must be exact and persisted.

Validation:
- KFold or GroupKFold depending discovered grouping;
- small-data-safe defaults.

Metrics:
- MAE;
- RMSE;
- R²;
- Spearman correlation;
- median absolute error;
- error by damage quantile.

## 13.5 Deep DamageNet

Architecture:

```text
stress/strain time-series
       ↓
multi-resolution TCN
       ↓
attention pooling
       ├──────────────┐
       ↓              ↓
neural features   physics features
       └────── fusion ┘
              ↓
         damage head
```

Output head:
- non-negative prediction using Softplus if target is guaranteed non-negative;
- optional mean + log-variance head for heteroscedastic regression.

## 13.6 Physics-aware loss

Only enable if sufficient physical information exists.

Potential form:

```text
L =
    L_data
  + λ_monotonic * L_monotonic
  + λ_physics * L_physics
```

Do not enforce arbitrary monotonicity between unrelated files unless a valid ordering/exposure variable exists.

## 13.7 SHM uncertainty

Implement:
- ensemble standard deviation;
- residual calibration;
- split conformal prediction interval where sample size permits.

Output:

```text
predicted_damage
lower_bound
upper_bound
uncertainty
```

Official submission writer can select only required fields.

## 13.8 SHM explanation

Show:
- top engineered fatigue proxies;
- influential time windows;
- stress-cycle histogram;
- predicted vs observed plots in evaluation;
- uncertainty interval.

---

# 14. Shared feature framework

Define a feature-extractor protocol:

```python
class FeatureExtractor(Protocol):
    def fit(self, samples, y=None): ...
    def transform(self, samples) -> pd.DataFrame: ...
    def fit_transform(self, samples, y=None) -> pd.DataFrame: ...
```

Feature names must be deterministic.

Persist:
- selected channels;
- scaling;
- fitted transforms;
- feature order.

No feature computation may use labels except explicitly supervised feature selection performed within training folds.

---

# 15. Shared model protocol

Define a high-level protocol so baseline/deep models can be used by common inference code.

Classification:

```python
fit(...)
predict(...)
predict_proba(...)
save(...)
load(...)
metadata()
```

Regression:

```python
fit(...)
predict(...)
predict_interval(...)  # if supported
save(...)
load(...)
metadata()
```

Deep models should expose:

```python
forward(x, mask=None, metadata=None)
encode(x, mask=None, metadata=None)
```

---

# 16. Self-supervised learning module

This is an advanced optional feature.

Do not auto-run it.

Implement:

## 16.1 Augmentations

Time-series augmentations:
- jitter;
- scaling;
- masking;
- crop;
- mild time warp;
- channel dropout.

All task-specific defaults must be conservative.

## 16.2 Masked reconstruction

Randomly mask spans/channels and reconstruct continuous sensor values.

## 16.3 Contrastive pretraining

Support positive views from two augmentations of the same window.

Use NT-Xent / InfoNCE.

## 16.4 Fine-tuning

Allow loading pretrained encoder weights into:
- DoorNet;
- CorrugationNet;
- DamageNet;
- ACV per-car encoder.

Do not force shared weights across physically incompatible sensor schemas. Share architecture or pretrained weights only when dimensions/semantics are valid.

---

# 17. Calibration

## Classification

Implement:
- temperature scaling for neural logits;
- isotonic calibration for classical models when enough calibration data exist;
- reliability diagrams;
- expected calibration error;
- Brier score.

Calibration must use held-out calibration predictions, not the same data used for fitting.

## Regression

Implement:
- residual interval calibration;
- optional split conformal intervals.

---

# 18. OOD detection

Implement at least two generic OOD methods:

1. Feature/embedding Mahalanobis distance.
2. k-nearest-neighbour embedding distance.

For deep models:
- fit OOD reference statistics on training embeddings only.

For classical feature models:
- use standardized engineered features.

Expose:

```text
ood_score
is_ood
```

Threshold chosen from training/validation quantile and stored in metadata.

OOD is a warning, not an automatic reclassification.

---

# 19. Robustness evaluation

Create corruption functions that operate on copies of sequences.

Required perturbations:

```text
gaussian_noise
amplitude_scale
constant_bias
random_point_dropout
contiguous_dropout
downsample_upsample
timestamp_jitter
single_channel_dropout
```

Each has severity levels.

Example config:

```yaml
robustness:
  gaussian_noise:
    severity: [0.01, 0.03, 0.05]
  random_point_dropout:
    severity: [0.05, 0.10, 0.20]
```

Report:
- clean metric;
- corrupted metric;
- retention ratio;
- absolute degradation;
- worst corruption;
- mean corruption score.

Never corrupt labels.

---

# 20. Explainability system

Create one unified explanation object:

```python
@dataclass
class Explanation:
    sample_id: str
    feature_importance: dict[str, float] | None
    channel_importance: dict[str, float] | None
    temporal_importance: np.ndarray | None
    spectral_importance: dict[str, float] | None
    neighbours: list[dict[str, Any]]
    notes: list[str]
```

Dashboard and report code should consume this object.

Avoid claiming causal explanations. Label UI as:
- "model evidence";
- "important features";
- "influential signal regions".

---

# 21. Health intelligence layer

This layer is **not** an official competition target. It is a demonstration/decision-support layer.

Define a normalized component health score:

```text
0 = worst / high concern
100 = healthiest / low concern
```

Inputs may include:
- task fault probability;
- anomaly score;
- damage estimate;
- uncertainty;
- OOD warning;
- persistence/trend if repeated observations exist.

The score formula must be configurable and transparent.

Example structure:

```text
risk =
    w_fault * calibrated_fault_risk
  + w_damage * normalized_damage_risk
  + w_anomaly * anomaly_risk

confidence_adjusted_risk =
    risk * confidence
    + conservative_prior * (1 - confidence)

health = 100 * (1 - clipped_risk)
```

Do not imply this is an LTA-approved maintenance rule.

Maintenance status labels for demo:

```text
Healthy
Monitor
Inspect Soon
High Priority
```

Thresholds configurable.

---

# 22. Experiment tracking

Every training/evaluation run should create:

```text
outputs/experiments/<experiment_id>/
    config_resolved.yaml
    metadata.json
    metrics.json
    fold_metrics.csv
    predictions.csv
    environment.json
    schema_hashes.json
    feature_names.json
    plots/
```

`metadata.json`:
- timestamp;
- task;
- model;
- seed;
- git commit;
- Python version;
- package versions;
- train/validation sizes;
- data manifest hash.

Do not depend on a hosted service.

---

# 23. Evaluation framework

## 23.1 Classification report

Return JSON-safe metrics and save plots.

Metrics:
- accuracy;
- balanced accuracy;
- macro precision;
- macro recall;
- macro F1;
- weighted F1;
- per-class metrics;
- confusion matrix;
- AUROC if valid;
- AUPRC if valid.

## 23.2 Regression report

Metrics:
- MAE;
- RMSE;
- R²;
- median absolute error;
- Spearman;
- optional MAPE only when denominator is safe.

## 23.3 Bootstrap confidence intervals

Implement bootstrap CIs at the **sample/file/case level**, never by treating highly correlated time points as independent samples.

## 23.4 Error analysis

Generate:
- hardest false negatives;
- hardest false positives;
- high-confidence errors;
- OOD errors;
- error by operation/class/target bin;
- relevant feature/signal snapshots.

---

# 24. Submission system

The official sample-submission files may be provided later.

Therefore implement a schema-driven submission layer.

Create configuration such as:

```yaml
submission:
  door:
    template: null
    id_column: null
    prediction_column: null
  acv:
    template: null
  corrugation:
    template: null
  shm:
    template: null
```

When an official example submission exists:
1. load it;
2. infer column names/order;
3. validate unique IDs;
4. preserve row order;
5. write predictions into the correct field;
6. reject missing/extra IDs;
7. preserve required data types.

Create:
- `generate_submission.py`;
- `validate_submission.py`;
- `packager.py`.

Packager outputs:

```text
outputs/submissions/predictions.zip
```

Never guess an official output schema if no template is available.

---

# 25. CLI requirements

All scripts should support `--help`.

Examples:

## Inspect

```bash
python scripts/inspect_data.py --task all
python scripts/inspect_data.py --task door
```

## Build manifests

```bash
python scripts/build_manifests.py --task all
```

## Build features

```bash
python scripts/build_features.py \
  --task door \
  --config configs/door/baseline.yaml
```

## Train baseline

```bash
python scripts/train_baseline.py \
  --task corrugation \
  --config configs/corrugation/baseline.yaml
```

## Deep training

```bash
python scripts/train_deep.py \
  --task door \
  --config configs/door/deep.yaml
```

## SSL

```bash
python scripts/pretrain_ssl.py \
  --task corrugation \
  --config configs/corrugation/dual_domain.yaml
```

## Evaluate

```bash
python scripts/evaluate.py \
  --task shm \
  --checkpoint outputs/checkpoints/... \
  --split validation
```

## Robustness

```bash
python scripts/robustness.py \
  --task door \
  --checkpoint ...
```

## Inference

```bash
python scripts/infer.py \
  --task acv \
  --checkpoint ... \
  --input data/raw/acv/test
```

## Submission

```bash
python scripts/generate_submission.py \
  --task all \
  --predictions-root outputs/predictions
```

```bash
python scripts/validate_submission.py \
  outputs/submissions/predictions.zip
```

No training script may execute simply from importing its module.

---

# 26. Training behavior

Training must be explicit.

Default behavior when the repository is first created:

```text
install
→ inspect
→ test
→ optionally build manifests/features
```

NOT:

```text
install → automatically train four deep networks
```

Training scripts should support:
- `--dry-run`;
- `--max-samples`;
- `--epochs`;
- `--device`;
- `--seed`;
- `--resume`;
- `--output-dir`.

`--dry-run` should:
- load data;
- build one batch;
- instantiate model;
- run one forward pass;
- compute one loss;
- exit.

This is critical for Codex validation without consuming GPU hours.

---

# 27. Deep training engine

Build a lightweight native PyTorch trainer, not a giant framework.

Features:
- CPU/GPU auto-selection;
- mixed precision on CUDA;
- gradient clipping;
- early stopping;
- scheduler support;
- checkpoint best/last;
- resume;
- deterministic seed;
- metric history;
- no hidden global state.

Losses:
- binary cross entropy;
- weighted cross entropy;
- focal loss;
- MSE;
- Huber;
- Gaussian NLL for uncertainty head.

---

# 28. Data leakage safeguards

Create explicit checks.

Examples:
- same source file hash cannot appear in train and validation;
- group IDs cannot overlap where group split is requested;
- scaler fit only on training fold;
- feature selector fit only on training fold;
- calibration set separate from fitting predictions;
- no label columns in feature table;
- submission test labels never loaded by training code.

Add tests for these safeguards.

---

# 29. Dashboard

Use Streamlit as the simplest default.

The dashboard must work in two modes:

```text
DEMO mode:
    uses synthetic/demo cached predictions

REAL mode:
    reads outputs produced by inference/evaluation
```

It must not require a trained checkpoint merely to start.

## 29.1 Main pages

### Fleet Overview

Display cards:
- total analysed components;
- faults detected;
- low-confidence predictions;
- OOD samples;
- mean health score.

Visual:
- simplified train/car layout;
- health status;
- clickable components.

### Door Diagnostics

Display:
- operation;
- predicted status;
- probability;
- uncertainty;
- motor current;
- voltage;
- EMF;
- position;
- highlighted salient region;
- nearest normal example.

### ACV Fault Localisation

Display:
- car ranking;
- probability per car;
- temperature traces;
- peer median;
- faulty-car candidate;
- confidence margin.

### Rail Corrugation

Display:
- predicted Normal / Side I / Side II;
- class probabilities;
- vibration trace;
- PSD/STFT;
- important channels;
- important frequency bands.

### SHM

Display:
- predicted cumulative damage;
- prediction interval;
- health score;
- stress trace;
- cycle histogram;
- influential intervals.

### Model Reliability

Display:
- calibration curve;
- clean vs corrupted performance;
- OOD scores;
- confidence/error scatter.

## 29.2 Visual quality

Avoid a default bare Streamlit appearance.

Use:
- clean page layout;
- consistent typography;
- metric cards;
- readable plots;
- restrained colors;
- explicit legends;
- no fake "live" sensor stream unless labelled simulation.

---

# 30. Optional API

If time permits, provide a small FastAPI extra, but do not make it mandatory.

Endpoints:

```text
GET /health
POST /predict/door
POST /predict/acv
POST /predict/corrugation
POST /predict/shm
```

Use Pydantic schemas.

This is lower priority than a correct competition pipeline.

---

# 31. Testing

## 31.1 Unit tests

Cover:
- config validation;
- door timestamp parser;
- segmentation boundaries;
- feature shape/order;
- manifest hashing;
- split leakage;
- model forward shapes;
- save/load;
- metrics;
- corruption functions;
- submission validation.

## 31.2 Synthetic integration test

Generate synthetic versions of all four tasks.

Door:
- normal and high-resistance-like cycles.

ACV:
- multiple cars with one car having slower thermal recovery.

Corrugation:
- three frequency-pattern classes.

SHM:
- signal amplitude/cycle severity correlated with synthetic damage.

End-to-end synthetic test:

```text
generate
→ manifest
→ features
→ tiny model fit
→ inference
→ evaluation
→ prediction write
```

Keep under a few seconds/minutes on CPU.

## 31.3 Tests must not depend on private raw data

Real-data integration tests should be separately marked.

---

# 32. Documentation

## README.md

Include:
1. problem;
2. architecture;
3. repository structure;
4. setup;
5. data placement;
6. schema inspection;
7. baseline training;
8. deep training;
9. evaluation;
10. dashboard;
11. submission;
12. known limitations.

Do not include invented scores.

## `docs/DATA_CONTRACTS.md`

Document the known schemas from Section 2 and discovered schemas.

## `docs/ARCHITECTURE.md`

Use Mermaid diagrams where useful.

## `docs/MODELLING.md`

Explain why each model is appropriate.

## `docs/EVALUATION.md`

Explain split methodology and metrics.

## `docs/ASSUMPTIONS.md`

Record every assumption, especially those made before raw data inspection.

## `docs/IMPLEMENTATION_STATUS.md`

Use checklist:

```text
[x] implemented
[~] partial
[ ] not implemented
```

Update as work progresses.

---

# 33. AGENTS.md

Create `AGENTS.md` summarising development rules for future Codex sessions.

Include:
- project mission;
- no fabricated schemas/results;
- no automatic expensive training;
- inspect before assumptions;
- tests required;
- raw data never committed;
- common commands;
- architecture boundaries;
- style conventions.

---

# 34. Code quality

Use:
- type hints;
- dataclasses/Pydantic;
- docstrings for public APIs;
- `pathlib`;
- Python logging, not scattered print statements;
- small cohesive functions;
- dependency injection where useful.

Avoid:
- giant 1000-line scripts;
- import-time file loading;
- notebook-only logic;
- hard-coded Windows paths;
- `sys.path` hacks;
- silent exception swallowing;
- global mutable config;
- hidden target leakage.

Ruff and pytest should pass.

Mypy may be configured pragmatically rather than requiring perfect strictness for third-party ML types.

---

# 35. Naming conventions

Use canonical task identifiers:

```text
door
acv
corrugation
shm
```

Canonical class names:

Door:

```text
normal
abnormal_resistance
```

Corrugation:

```text
normal
side_i
side_ii
```

Preserve official label spelling at submission boundaries.

---

# 36. Model selection logic

Implement a model registry.

Example:

```python
MODEL_REGISTRY = {
    "door": {
        "logreg": ...,
        "extra_trees": ...,
        "doornet": ...,
        "ensemble": ...,
    },
    ...
}
```

Model metadata must record:
- task;
- feature schema;
- input channels;
- class ordering;
- preprocessing version;
- checkpoint version.

Inference must reject incompatible schemas rather than silently reordering unknown inputs.

---

# 37. Ensembling

Build generic ensembling.

Classification:
- probability averaging;
- weighted averaging;
- optional stacking trained out-of-fold.

Regression:
- weighted mean;
- inverse-validation-error weighting;
- uncertainty from ensemble spread.

Never fit stacking meta-model on in-sample base predictions.

Default competition strategy should support combining:
- strong engineered-feature model;
- deep time-series model;
- task-specific anomaly/physics model.

---

# 38. Recommended competition model strategy

Do not assume the fanciest network wins.

Recommended progression:

## Door
1. engineered ExtraTrees/boosting;
2. operation-specific baseline;
3. compact DoorNet;
4. normality score;
5. calibrated ensemble.

## ACV
1. relative-rule score;
2. relational engineered classifier;
3. leave-one-case-out validation;
4. only then cross-car attention model.

## Corrugation
1. spectral/statistical boosting;
2. class-weighted baseline;
3. dual-domain network;
4. ensemble.

## SHM
1. fatigue/stress feature baseline;
2. boosted regressor;
3. physics-feature + TCN;
4. uncertainty-aware ensemble.

---

# 39. Hyperparameter tuning

Support Optuna optionally.

Do not auto-run a large sweep.

Command should require explicit invocation.

Use:
- nested or fold-correct evaluation;
- pruning;
- bounded search spaces;
- stored study results.

Small data means broad 500-trial sweeps are usually inappropriate.

---

# 40. Ablation system

Allow comparison of:
- time only;
- spectral only;
- time + spectral;
- no channel attention;
- no engineered features;
- no calibration;
- no normality branch;
- no physics features.

Produce a CSV suitable for final presentation.

---

# 41. Operational inference package

Provide a high-level API:

```python
from railguard.inference import RailGuardPredictor

predictor = RailGuardPredictor.from_bundle("path/to/bundle")

result = predictor.predict(
    task="door",
    input_path="..."
)
```

Model bundle should include:
- weights/model;
- preprocessing;
- channel schema;
- config;
- class map;
- calibration;
- OOD statistics;
- metadata.

---

# 42. Model bundle format

Use a directory, not one opaque pickle:

```text
bundle/
├── metadata.json
├── config.yaml
├── schema.json
├── preprocessor.joblib
├── model.joblib            # classical
# OR
├── model.pt                # PyTorch
├── calibration.joblib
├── ood.joblib
└── feature_names.json
```

Avoid unsafe loading from untrusted locations.

---

# 43. Data placement instructions for the user

Document expected layout:

```text
data/raw/
├── door/
│   ├── Train.csv
│   ├── Test.csv
│   ├── Train_Segments_Answer.csv
│   └── ... official submission templates if supplied
│
├── acv/
│   ├── Train_Labels.csv
│   ├── train/
│   │   ├── acv_case_01.xlsx
│   │   └── ...
│   ├── test/
│   └── ...
│
├── corrugation/
│   ├── Train_Labels.csv
│   ├── train/
│   │   ├── Train1.csv
│   │   └── ...
│   ├── test/
│   └── ...
│
└── shm/
    ├── Train_Labels.csv
    ├── train/
    │   ├── train01.csv
    │   └── ...
    ├── test/
    └── ...
```

If actual official layout differs, adapters/config should support it without requiring users to rename thousands of files.

---

# 44. `.gitignore`

At minimum:

```gitignore
.env
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/

data/raw/**
data/interim/**
data/processed/**
!data/README.md

outputs/checkpoints/**
outputs/experiments/**
outputs/predictions/**
outputs/submissions/**

*.pt
*.pth
*.ckpt
*.joblib
*.pkl
```

Keep small test fixtures trackable under `tests/fixtures/`.

---

# 45. Security / secrets

Never commit:
- API keys;
- passwords;
- absolute machine paths.

Use `.env.example`.

This PS3 project does not require LTA DataMall for core modelling. If a future demo integrates DataMall, isolate it under an optional integration module and keep credentials out of source control.

---

# 46. What NOT to do

Do not:

- train on test data;
- infer test labels;
- use random row splitting on one continuous sensor file;
- report accuracy alone for corrugation;
- train a giant Transformer on 110 door segments and call it scientific;
- pretend six ACV cases justify a high-capacity classifier;
- invent S-N curve constants;
- call correlation causation;
- hard-code a 20 ms sample interval without validation;
- hard-code ACV workbook schema before inspecting it;
- hard-code official submission format without sample templates;
- force all four datasets into one identical representation;
- silently drop malformed samples;
- silently impute huge missing regions;
- write results that were not generated;
- auto-launch GPU training from setup or tests.

---

# 47. Phased implementation order

Codex should execute these phases in order.

## Phase 1 — Project foundation

Create:
- repository structure;
- `pyproject.toml`;
- `.gitignore`;
- config system;
- logging;
- common types;
- docs skeleton;
- `AGENTS.md`.

Acceptance:
- package imports;
- `pytest` runs;
- `ruff check` passes or has only documented exclusions.

## Phase 2 — Data inspection and manifests

Implement:
- schema inspector;
- file hashing;
- manifest;
- base adapter;
- task registry;
- `inspect_data.py`;
- `build_manifests.py`.

Acceptance:
- works on synthetic fixtures;
- can inspect CSV and XLSX;
- does not need full data in memory for basic inspection.

## Phase 3 — Door data adapter

Implement:
- timestamp parser;
- label reading;
- segment slicing;
- row-count validation;
- test segmentation fallback;
- synthetic tests.

Acceptance:
- supplied known door schema is supported;
- exact segment matching diagnostics exist.

## Phase 4 — Generic signal processing and features

Implement:
- cleaning;
- derivatives;
- FFT/PSD/STFT;
- optional wavelets;
- statistical/temporal/spectral features.

Acceptance:
- deterministic feature names;
- no NaN/Inf leakage without explicit handling.

## Phase 5 — Classical baseline framework

Implement:
- classification/regression wrappers;
- pipelines;
- split logic;
- metrics;
- model serialization;
- baseline CLI.

Acceptance:
- tiny synthetic training works for all four task types.

## Phase 6 — Door baseline

Implement:
- door engineered features;
- cross-validation;
- reports;
- inference;
- explanations.

Do not perform expensive tuning.

## Phase 7 — ACV adapter + relational baseline

Implement:
- Excel schema discovery;
- per-car extraction abstraction;
- relative features;
- candidate ranking;
- leave-one-case-out evaluator.

If real schema absent, ensure clear configuration path and synthetic fixture.

## Phase 8 — Corrugation adapter + baseline

Implement:
- file adapter;
- vibration features;
- imbalance-aware evaluation;
- macro-F1 reporting.

## Phase 9 — SHM adapter + baseline

Implement:
- schema mapping;
- stress/fatigue proxy features;
- rainflow support;
- regression metrics;
- target transforms.

## Phase 10 — Deep shared backbones

Implement:
- CNN1D;
- TCN;
- Transformer;
- spectral encoder;
- channel attention;
- fusion;
- mask-aware pooling.

Acceptance:
- forward-shape tests;
- variable sequence-length tests.

## Phase 11 — Task-specific deep models

Implement:
- DoorNet;
- ACV relational attention model;
- Corrugation dual-domain model;
- SHM DamageNet.

No full training.

Run synthetic forward/backward smoke tests.

## Phase 12 — Training engine

Implement:
- trainer;
- early stopping;
- AMP;
- checkpoints;
- resume;
- dry-run.

Acceptance:
- one synthetic epoch passes.

## Phase 13 — SSL

Implement:
- augmentations;
- masked reconstruction;
- contrastive objective;
- encoder export.

No pretraining automatically.

## Phase 14 — Calibration and uncertainty

Implement:
- temperature;
- isotonic hook;
- conformal/regression interval;
- ensemble uncertainty;
- OOD score.

## Phase 15 — Explainability

Implement:
- tree feature importances;
- permutation;
- temporal saliency;
- Captum optional;
- neighbour retrieval;
- spectral attribution.

## Phase 16 — Robustness

Implement corruption suite and reporting.

## Phase 17 — Health intelligence

Implement:
- score;
- trend;
- maintenance status;
- transparent configuration.

## Phase 18 — Submission system

Implement template-driven submission generation/validation.

Do not guess official schemas.

## Phase 19 — Dashboard

Implement demo + real modes.

Acceptance:
- app starts without trained models;
- synthetic demo data render;
- no fake performance claims.

## Phase 20 — Final integration

Add:
- end-to-end synthetic test;
- README commands;
- Makefile targets;
- implementation-status update;
- architecture diagram;
- final import/lint/test pass.

---

# 48. Makefile

Suggested targets:

```makefile
install
install-dev
test
lint
format
inspect
manifest
dashboard
smoke
clean
```

Do not make `all` train models.

---

# 49. Final acceptance criteria

The repository is complete when all of the following are true.

## Data

- [ ] All four adapters exist.
- [ ] Door known schema works.
- [ ] ACV schema can be discovered from XLSX.
- [ ] Corrugation file schema can be discovered.
- [ ] SHM file schema can be discovered.
- [ ] Manifests can be built.
- [ ] Leakage checks exist.

## Features

- [ ] Statistical features implemented.
- [ ] Temporal features implemented.
- [ ] Spectral features implemented.
- [ ] Wavelet features optional.
- [ ] Task-specific features implemented.
- [ ] Feature schema persisted.

## Models

- [ ] Strong baseline classifier/regressor framework.
- [ ] Door baseline + DoorNet + normality model.
- [ ] ACV relational baseline + attention model.
- [ ] Corrugation baseline + dual-domain model.
- [ ] SHM baseline + DamageNet.
- [ ] Ensemble support.
- [ ] Save/load bundles.

## Reliability

- [ ] Calibration support.
- [ ] Uncertainty support.
- [ ] OOD scoring.
- [ ] Robustness corruptions.
- [ ] Explainability.

## Engineering

- [ ] Typed package.
- [ ] YAML configuration.
- [ ] Tests.
- [ ] Linting.
- [ ] Dry-run training.
- [ ] Experiment logging.
- [ ] No auto training.
- [ ] No raw data committed.

## Submission

- [ ] Template-driven writer.
- [ ] Validator.
- [ ] ZIP packager.
- [ ] Missing/duplicate ID checks.

## Demo

- [ ] Streamlit dashboard.
- [ ] Demo mode.
- [ ] Real-output mode.
- [ ] Four task views.
- [ ] Reliability view.
- [ ] Health view.

## Documentation

- [ ] Complete README.
- [ ] Architecture document.
- [ ] Data contracts.
- [ ] Modelling rationale.
- [ ] Evaluation methodology.
- [ ] Submission guide.
- [ ] Assumptions.
- [ ] Implementation status.

---

# 50. Final deliverables Codex should leave behind

When implementation is finished, the repository should allow the user to do approximately:

```bash
# 1. Install
pip install -e ".[all]"

# 2. Inspect actual raw data
python scripts/inspect_data.py --task all

# 3. Build manifests
python scripts/build_manifests.py --task all

# 4. Smoke-test everything
pytest -q
python scripts/train_deep.py --task door --config configs/door/deep.yaml --dry-run

# 5. Explicitly train later, only when user chooses
python scripts/train_baseline.py --task door --config configs/door/baseline.yaml
python scripts/train_baseline.py --task acv --config configs/acv/baseline.yaml
python scripts/train_baseline.py --task corrugation --config configs/corrugation/baseline.yaml
python scripts/train_baseline.py --task shm --config configs/shm/baseline.yaml

# 6. Train advanced models later
python scripts/train_deep.py --task door --config configs/door/deep.yaml
python scripts/train_deep.py --task acv --config configs/acv/relational.yaml
python scripts/train_deep.py --task corrugation --config configs/corrugation/dual_domain.yaml
python scripts/train_deep.py --task shm --config configs/shm/deep.yaml

# 7. Evaluate / robustness / explainability
python scripts/evaluate.py --task door --checkpoint <bundle>
python scripts/robustness.py --task door --checkpoint <bundle>
python scripts/explain.py --task door --checkpoint <bundle>

# 8. Inference
python scripts/infer.py --task all --checkpoint-root outputs/checkpoints

# 9. Generate and validate official submission after templates are supplied
python scripts/generate_submission.py --task all
python scripts/validate_submission.py outputs/submissions/predictions.zip

# 10. Demo
streamlit run dashboard/app.py
```

---

# 51. Closing implementation instruction to Codex

Build this as a serious ML engineering project.

The order of priority is:

```text
correct data handling
> leakage-free evaluation
> strong baselines
> robust competition outputs
> advanced models
> uncertainty/explainability
> polished dashboard
```

Do not sacrifice correctness for architectural novelty.

Where the data are small, prefer statistically defensible models and use advanced neural components as complementary experiments rather than automatically assuming they are superior.

Where the data schema is unknown, inspect it and create an explicit mapping rather than guessing.

Where official submission templates are missing, build a template-driven validator and wait for the real template rather than inventing one.

The final codebase should be runnable, modular, tested, reproducible, and ready for the user to place the full NebulaX PS3 datasets into `data/raw/` and begin experiments manually.
