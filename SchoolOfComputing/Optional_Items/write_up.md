# RailGuard AI: Train Condition Monitoring Across Four Subsystems

## 1. Overview

RailGuard AI covers all four NebulaX PS3 tracks: Door operation detection and
abnormal-resistance classification, ACV refrigerant-leak localisation, Rail
Corrugation classification, and Structural Health Monitoring (SHM)
cumulative-fatigue-damage estimation.

The tracks have different inputs, outputs, and official scoring rules, so we
developed separate task-specific feature pipelines and models behind one shared
web application. Our priorities were reproducibility, leakage-aware validation,
preservation of official identifiers, and outputs that non-technical users can
inspect and download.

We inspected the supplied schemas before implementing each adapter and retained
the raw data unchanged. Feature extraction and model selection use deterministic
seed 42. Validation occurs at the natural prediction unit--door operation, ACV
case, or complete source file--rather than by splitting correlated signal rows.

## 2. Modelling approach

### 2.1 Door monitoring

Door processing first detects opening and closing operations in the continuous
stream, then classifies each operation as `Normal` or `Abnormal resistance`.
Test segmentation is label-free and uses timestamp discontinuities plus command
and state channels. Short fragments are rejected, nearby active regions can be
merged, and original timestamps are retained as submission boundaries.

Features describe motor current and voltage, door travel, velocity,
acceleration, stall fraction, monotonicity violations, current per unit of
position change, power and energy proxies, current-velocity correlation and
lag, and operation direction. Statistical, temporal, and spectral summaries are
also extracted per channel.

The classifier averages probabilities from balanced Extra Trees, Random Forest,
and histogram gradient boosting models. Tree ensembles suit nonlinear feature
interactions and the small labelled dataset. The training annotations contain
110 operations: 80 `Normal` and 30 `Abnormal resistance`, evenly split into 55
opening and 55 closing operations.

Five-fold classifier validation is stratified jointly by operation and status.
The saved metadata records an out-of-fold **classification macro F1 of 1.0000**
on labelled ground-truth segments. This does not evaluate inferred boundaries
and must not be presented as the official end-to-end score. The official Door
metric is IoU-weighted F1 because correct labels and accurate boundaries are
both necessary: misses reduce soft recall, extra segments reduce soft precision,
and inaccurate boundaries receive partial IoU credit. The supplied Door bundle
did not include its detailed fold report, so none was invented.

### 2.2 ACV refrigerant-leak localisation

ACV is treated as a within-case ranking problem rather than direct eight-class
classification because only six labelled cases are available and the workbooks
contain two vendor schemas.

For each car, 23 absolute summaries cover cabin, outdoor, and target temperature
behaviour; distribution statistics and slopes; target error; outdoor-to-cabin
gap; operating-state changes; and missingness. The same summaries are expressed
relative to the median car in the case, producing 46 absolute and peer-residual
features. Peer residuals control for conditions shared by the train.

The selected ranker combines a balanced logistic-regression probability with a
robust peer-anomaly score. Both are converted to within-case percentile ranks:

`score = 0.5 * supervised_percentile + 0.5 * peer_anomaly_percentile`

Leave-one-case-out validation holds out all cars from one workbook together and
achieved a mean linear rank-decay score of **0.9792** across six cases. This is
a validation estimate, not hidden-test performance. Rank decay suits the output
because it rewards placing the true faulty car near the top of the inspection
list while retaining partial credit for useful shortlists.

### 2.3 Rail corrugation classification

Each vibration file is classified as `Normal`, `Side I`, or `Side II`. The
training distribution is imbalanced: 234 `Normal`, 14 `Side I`, and 24 `Side II`.

The extractor produces 85 features from the documented 10 kHz vibration and
shock measurements. They include RMS, spread, peaks, skewness, kurtosis, crest
factor, zero crossings, dominant frequency, spectral centroid and entropy,
frequency-band ratios, estimated speed, speed-normalised spatial frequencies,
and differences between sides.

Balanced logistic regression, Extra Trees, Random Forest, and histogram
gradient boosting were compared using five-fold stratified cross-validation.
The selected balanced histogram gradient-boosting model uses 250 iterations,
learning rate 0.05, at most seven leaves, at least 20 samples per leaf, and L2
regularisation 1.0. Its validation results were **0.8714 macro F1** and **0.8798
balanced accuracy**. Macro F1 is appropriate because each class contributes
equally and a majority-only classifier cannot hide failures on rare faults.

### 2.4 SHM fatigue-damage estimation

SHM predicts one positive cumulative-damage value per stress-history file. Its
53 features include distribution statistics, RMS and energy, peaks and
excursions, derivatives, rainflow cycle counts, cycle-range statistics,
weighted quantiles, normalised histograms, and fatigue proxy sums using
exponents from 2 through 8.

Material S-N constants are not disclosed, so we do not claim to calculate
physical Miner damage directly. The exponent bank provides generic cycle
severity proxies from which the model learns the relationship in the labels.

The selected estimator is standardised Ridge regression on `log(damage)`. This
handles the broad positive target range and regularises correlated fatigue
features across 64 labelled files. Ridge, Extra Trees, Random Forest, and
histogram gradient boosting were compared with raw, `log1p`, and log targets
using five-fold validation stratified over binned target ranks.

Selected validation results were **0.9614 MAPE-derived score**, **3.86% MAPE**,
**0.00856 MAE**, and **0.9966 Spearman correlation**. The organiser-defined
score `max(0, 1 - MAPE)` is suitable because proportional error remains
interpretable across different damage magnitudes.

## 3. Methodological safeguards

- Raw files remain immutable.
- Door test segmentation uses no status labels.
- ACV holds out complete cases.
- Corrugation and SHM hold out complete files, never signal rows.
- Preprocessing is fitted within model pipelines.
- Feature order and schemas are persisted with each bundle.
- Inference rejects incompatible task or feature schemas.
- Official filenames, labels, timestamps, and leading-zero identifiers are
  preserved.
- Random seeds and selected configurations are recorded.

Reported cross-validation results were used for model selection and are
comparative estimates, not results from a separate untouched holdout set.

## 4. Assumptions and open decisions

- The observed Door interval is approximately 20 ms (50 Hz); timestamps are
  parsed and checked rather than overwritten.
- Door command and state channels are sufficient for label-free segmentation.
- Door operations come from one continuous training stream with no higher-level
  group identifier, so operation/status stratification may not capture every
  temporal dependency.
- ACV identifiers and parameters are discovered per workbook because two vendor
  schemas are present.
- Cars in an ACV case share enough context for peer-median residuals.
- Corrugation uses the documented 10 kHz rate and speed-sensor/wheel geometry.
  With no extra grouping metadata, stratified file-level folds are used.
- SHM file numbers are identifiers rather than temporal order, and positive
  targets permit logarithmic modelling.
- No S-N curve or material constants were invented.
- App health messages are decision support, not approved maintenance rules.

## 5. Application and explainability

The React and FastAPI app lets non-technical users select a subsystem, upload a
supported file, inspect the prediction, and download the official CSV. Results
use progressive disclosure: a quick summary, supporting evidence, technical
probabilities/rankings/intervals, and educational guidance. Model evidence is
not presented as physical causation, and demonstration outputs are distinguished
from trusted-model predictions.

## 6. Development code and models

The accompanying task folders contain relevant adapters, feature extraction,
models, official metrics, selection logic, inference entry points, and
configurations. Model folders include serialized estimators, preprocessing,
feature order, schema, configuration, metadata, and detailed CV reports where
available. Bundles are included for all four tasks; Door's shifted source
filenames were normalized and documented. No raw dataset is included.

## 7. Reproducibility

Inspect and manifest the supplied data before explicit training:

```text
python scripts/inspect_data.py --task all --raw-root <PS3/02_Datasets>
python scripts/build_manifests.py --task all --raw-root <PS3/02_Datasets>
python scripts/train_competition.py --task all --raw-root <PS3/02_Datasets>
```

Verification commands are `pytest -q` and `ruff check .`.

