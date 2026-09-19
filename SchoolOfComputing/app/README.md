# RailGuard AI

This directory is the self-contained source-code package for the RailGuard application. Run build,
test and deployment commands from this directory so the existing relative paths and Docker build
contexts continue to work.

The package contains the React frontend, FastAPI backend, shared `railguard` inference library,
configuration, operational scripts, deployment definitions, documentation and tests. Generated
predictions are submitted separately. Model checkpoints, raw/processed data, local databases,
installed dependencies, caches and compiled frontend output are intentionally excluded.

For real inference, provide the approved model bundles separately and set the corresponding
`RAILGUARD_DOOR_BUNDLE`, `RAILGUARD_ACV_BUNDLE`, `RAILGUARD_CORRUGATION_BUNDLE` and
`RAILGUARD_SHM_BUNDLE` paths. The Google Cloud deployment can continue to point these variables at
the Cloud Storage volume mounted at `/models`.

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

For the small-data competition tracks, use metric-aligned model selection instead of assuming a
deep network will generalise:

```bash
python scripts/train_competition.py --task all \
  --raw-root /path/to/NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets
```

This explicitly performs Door operation/status-stratified ensemble validation, ACV leave-one-case-
out rank selection, stratified Rail macro-F1 model selection, and SHM log-damage/rainflow model
selection by MAPE. It caches read-only engineered
features under `data/processed/competition` and writes trusted bundles under
`outputs/checkpoints/competition`. On the supplied training data with seed 42, the selected
out-of-fold scores were ACV 0.9792 rank decay, Rail 0.8714 macro F1, and SHM 0.9614 MAPE-derived
score. These are validation estimates, not claims about hidden-test performance.

## Models used for the competition predictions

The prediction files in `outputs/predictions` use CPU-based classical machine-learning models.
They do not use the optional PyTorch neural networks. Each task has a different output and scoring
rule, so one universal model would be a poor fit. Feature transformations and model preprocessing
are learned from training folds only during model selection.

### Door: operation segmentation and a tree ensemble

**Goal:** find every door movement in one continuous signal and label the movement `Normal` or
`Abnormal resistance`.

1. The continuous stream is divided into door operations using timestamp gaps and the open/close
   command and state channels. The original timestamps become the submitted segment boundaries.
2. Each operation is converted into statistics describing motor current, voltage, force, door
   travel, velocity, acceleration, stalls, energy, and temporal/spectral behaviour.
3. Three classifiers are fitted to the 110 labelled training operations: Extra Trees, Random
   Forest, and histogram gradient boosting. The first two average many randomized decision trees;
   gradient boosting builds small trees sequentially, with each tree correcting errors made by the
   trees before it.
4. The three class-probability vectors are averaged, and the class with the highest mean
   probability is written to `door_predictions.csv`.

In compact form, the final decision is
`argmax_class((P_extra_trees + P_random_forest + P_gradient_boosting) / 3)`. The separate boundary
detection stage matters because the Door metric evaluates both segment overlap and classification.
The fitted ensemble is stored under `outputs/checkpoints/competition/door`. Door predictions are
generated by reloading that bundle, so the CSV can be reproduced from the supplied `Test.csv`.

### ACV: supervised probability plus peer anomaly ranking

**Goal:** rank all eight cars in one Excel case from most to least likely to have a refrigerant
leak.

Every car is summarized using its temperature, error, running-state, change, and stability
statistics. The same car is also compared with the other seven cars in that case. These
peer-residual features describe how unusually that car behaves relative to the fleet under the
same operating conditions.

The selected model combines two signals with equal weight:

- a balanced logistic-regression probability learned from labelled cars; and
- a robust anomaly score based on the median absolute peer residual.

Both signals are converted to percentile ranks within the current case before they are averaged:
`score = 0.5 * supervised_percentile + 0.5 * peer_anomaly_percentile`. Cars are sorted by this
score, producing a string such as `01|04|08|03|05|07|02|06`. Model type and mixture weight were
selected using leave-one-case-out validation, so a validation case was never used to train the
model that ranked it.

### Rail corrugation: balanced histogram gradient boosting

**Goal:** classify each vibration file as `Normal`, `Side I`, or `Side II`.

The feature extractor summarizes time-domain vibration shape, frequency-domain energy and peaks,
speed transitions, speed-normalized spatial frequencies, and differences between the two sides.
The selected classifier is a class-balanced histogram gradient-boosting model with 250 boosting
iterations, learning rate 0.05, at most 7 leaves per tree, at least 20 samples per leaf, and L2
regularization of 1.0.

Histogram boosting first groups continuous feature values into bins. It then builds a sequence of
small decision trees; each new tree reduces the classification loss left by the existing ensemble.
Class balancing makes mistakes on the uncommon `Side I` and `Side II` examples count more during
training. The configuration was selected by five-fold stratified cross-validation using macro F1,
which gives each of the three classes equal importance.

### SHM: rainflow features and log-target Ridge regression

**Goal:** predict one positive cumulative fatigue-damage value for each stress-history file.

Rainflow counting converts the stress trace into load cycles with different ranges and counts.
The model inputs include weighted cycle-range quantiles, normalized cycle histograms, general
time/frequency statistics, and damage-proxy sums for exponent values from 2 through 8. These are
data-driven proxies only: the code does not invent an S-N curve or material constant that was not
provided in the problem data.

The target spans a large range, so training uses `z = log(damage)`. A standardized Ridge regression
then learns a coefficient for every feature while applying an L2 penalty to prevent unstable large
coefficients:
`minimize ||z - Xw||^2 + alpha * ||w||^2`. At inference, the prediction is transformed back with
`exp(z_hat)` and multiplied by the cross-validation-derived factor `0.997811`. The model family,
target transform, and scale factor were selected with five-fold validation using the official
MAPE-derived score. Its out-of-fold MAPE was approximately 3.86%; this is a validation estimate,
not a guarantee of the hidden-test score.

### Optional deep-learning models

The repository also contains compact temporal convolution, spectral encoding, channel attention,
and attention-pooling networks for experimentation. They repeatedly transform learned sequence
representations and can use GPU acceleration, but they were not used to create the four CSVs in
the current `predictions.zip`. With these relatively small labelled datasets, the validated
classical models were selected because they generalized better and were cheaper to train.

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

## React web app and compulsory workflow

The primary operator app is a responsive React interface backed by FastAPI. It covers all four
subsystems, validates file types, runs configured trusted bundles through the existing Python
predictor, renders task-specific results, and downloads the exact official CSV schema. With no
bundle configured it can use an explicitly labelled demonstration fallback that cannot be
mistaken for a submission result.

The local operator workflow extracts available asset ID, component coverage, and measurement time
as soon as a file is selected, then lets the operator correct them before analysis. ACV metadata is
read directly from the workbook. Door has an embedded timestamp but no train ID; Rail and SHM have
neither, so their filename and file-modification time are used as visibly labelled fallbacks rather
than fabricating a train number. A confirmed result can be saved to `data/railguard.db`. SQLite
history stores result data, model version, mode,
component/location metadata, and the upload's SHA-256 hash without duplicating raw sensor files.
Predictions are immutable after saving; asset metadata remains editable. The History workspace
provides filters, asset summaries, task-specific trends, chronological records, and saved-result
inspection. Demo records are supported but remain explicitly labelled.

```bash
pip install -e ".[api]"
npm --prefix web/frontend install
make api  # terminal 1
make web  # terminal 2
```

Open `http://localhost:5173`. Configure bundle paths with the `RAILGUARD_*_BUNDLE` environment
variables documented in [`web/README.md`](web/README.md), or run both services with
`docker compose up --build`.

### Operator decision UI

The result screen translates task-specific model output into two levels of information: a fast
decision summary for operators and progressively disclosed evidence for engineers. It does not
turn a prediction into an automatic maintenance instruction.

The default **Quick decision** view displays:

- **Status:** `Healthy`, `Monitor`, `Inspect Soon`, or `High Priority`;
- **Location:** the reported car, door, rail side, segment, or structural component, with an
  explicit generic fallback when the input does not provide an identifier;
- **Finding:** a plain-language description of the model output;
- **Urgency:** when the configured playbook says the finding should be reviewed;
- **Next checks:** a non-authoritative checklist from the configured playbook; and
- **Reliability:** `Reliable`, `Review advised`, or `Insufficient evidence`, together with the
  reason for that label.

Every checklist is labelled **Decision support** in the interface. Checklists come from the typed
`DecisionPlaybook` in `web/frontend/src/decision.ts`, not from free-form model output. The bundled
values are conservative demonstration defaults and must be reviewed or replaced with an
organisation-approved playbook before operational use. A custom playbook can be passed explicitly:

```ts
import {
  buildDecision,
  DEFAULT_DECISION_PLAYBOOK,
  type DecisionPlaybook,
} from "./decision";

const approvedPlaybook: DecisionPlaybook = structuredClone(DEFAULT_DECISION_PLAYBOOK);
approvedPlaybook.door.abnormal = {
  status: "High Priority",
  urgency: "Apply the approved depot response procedure now",
  checks: ["Use the approved abnormal-resistance checklist."],
};

const decision = buildDecision(prediction, approvedPlaybook);
```

The remaining result views provide progressive disclosure:

1. **Why this result?** gives two or three plain-language evidence statements and explains the
   reliability label without claiming a physical cause.
2. **Technical evidence** shows the task-specific cycle table, car ranking, class probabilities,
   uncertainty, or damage interval available in the API response.
3. **Learn** defines the relevant parameter, explains how to read its chart, distinguishes model
   influence from causation, and only refers to normal ranges when an approved source supplies
   them.

ACV results additionally include a train-car inspection map and top-two score margin. ACV bars use
a fixed 0–1 scale so a weak leading candidate is not visually inflated. SHM damage is shown as a
raw estimate and interval rather than an invented percentage or pass/fail threshold. Demonstration
results are always labelled `Insufficient evidence` and cannot support operational action.

Run the UI regression gates with:

```bash
npm --prefix web/frontend test
npm --prefix web/frontend run build
npm --prefix web/frontend run lint
```

## Legacy Streamlit dashboard

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
