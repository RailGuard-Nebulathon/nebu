# Implementation status

Legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented.

- [x] Phase 1 — project foundation (validated config, package types, logging, configs, docs and tooling)
- [x] Phase 2 — bounded-memory CSV/XLSX inspection, SHA-256 manifests, registry, leakage validation and CLIs
- [x] Phase 3 — robust timestamps, exact labelled slicing diagnostics and label-free test segmentation
- [x] Phase 4 — cleaning, derivatives, filtering/resampling, FFT/PSD/STFT, optional wavelets and deterministic features
- [x] Phase 5 — typed classical pipelines, sample/group splits, metrics, bundles, registry and baseline CLI
- [x] Phase 6 — operation/physics features, fold reports, normality model, explanations-ready evidence and official IoU scorer
- [x] Phase 7 — XLSX discovery/mapping, per-car extraction, relative candidates, rule ranking and leave-one-case-out evaluation
- [x] Phase 8 — 129-column adapter, side-aware vibration/spectral features and imbalance-aware macro-F1 baseline
- [x] Phase 9 — headerless stress adapter, rainflow/fatigue proxies, target transforms and official MAPE score
- [x] Phase 10 — multiscale CNN, residual TCN, Transformer, spectral encoder, channel attention, fusion and mask-aware pooling
- [x] Phase 11 — DoorNet, ACV relational attention/graph, CorrugationNet and heteroscedastic DamageNet
- [x] Phase 12 — deterministic trainer, AMP, clipping, scheduler, early stopping, checkpoints/resume and dry-run
- [x] Phase 13 — conservative augmentations, masked reconstruction, NT-Xent and encoder export (opt-in only)
- [x] Phase 14 — temperature/isotonic calibration, conformal intervals, ensemble uncertainty and two OOD scores
- [x] Phase 15 — tree/permutation evidence, temporal/channel saliency, optional Captum, neighbours and spectral attribution
- [x] Phase 16 — eight copy-safe corruptions with severity and clean/retention/degradation reporting
- [x] Phase 17 — configurable confidence/OOD-adjusted health, robust trend and maintenance status
- [x] Phase 18 — exact official schemas, template/ID validation, flat ZIP packaging and missing/duplicate checks
- [x] Phase 19 — styled Demo/Real Streamlit app with upload/download, four tracks, reliability and health views
- [x] Phase 20 — operational predictor, experiment artifacts, synthetic E2E, complete README/commands and final audit

## Final acceptance audit

### Data and features

- [x] All four adapters; known Door schema; ACV XLSX, corrugation and SHM discovery; hashed manifests.
- [x] File/group leakage checks and target-column rejection.
- [x] Statistical, temporal, spectral, optional wavelet and all task-specific features.
- [x] Deterministic feature order persisted in model bundles.

### Models and reliability

- [x] Classical classifier/regressor framework with sklearn fallback.
- [x] Door baseline, DoorNet and normality model.
- [x] ACV relational baseline, native attention and graph layer.
- [x] Corrugation baseline and dual-domain model.
- [x] SHM fatigue baseline and DamageNet.
- [x] Probability/regression ensembling, OOF stacking guard, transparent save/load bundles.
- [x] Calibration, uncertainty, OOD, corruption robustness and unified explainability components.

### Engineering and submissions

- [x] Typed package, validated YAML, deterministic seeds, dry-run trainer and local experiment records.
- [x] Tests, compile/import checks, no automatic training, ignored raw/generated artifacts.
- [x] Template-driven writer, strict validator, flat ZIP packager, missing/duplicate ID checks.
- [x] Demo/Real dashboard, uploads/downloads, four task views, reliability and health views.
- [x] Responsive React operator app and FastAPI inference boundary for all four official upload/output workflows.
- [x] Reproducible Door ensemble bundle, metric-aligned ACV leave-one-case-out ranker, speed-aware Rail model selection, and full-signal SHM fatigue regression.
- [x] README, architecture, data contracts, modelling, evaluation, submission, dashboard, assumptions and status docs.

## External/user actions that are intentionally not fabricated

- [ ] Train/select final checkpoints and generate held-out predictions (explicit user action; expensive training prohibited in this build).
- [ ] Record the compulsory ≤3-minute app demo video (human/external artifact).
- [ ] Package the final team-named submission folder after the team name and selected models are known.
- [~] Ruff configuration is complete; Ruff is not installed in this execution environment, so compile/import and pytest gates were used here.

## Post-acceptance hardening

- [x] ACV deep-sequence construction now canonicalizes the two supplied vendor schemas, encodes categorical operating states, records source-column/category metadata, appends per-signal availability masks, and masks empty template-car blocks instead of requiring an impossible exact numeric-column intersection.
- [x] `better-results` now presents a playbook-backed quick decision with Healthy/Monitor/Inspect Soon/High Priority status support, reported component location, plain-language finding, urgency, next checks and Reliable/Review advised/Insufficient evidence states.
- [x] Result disclosure is separated into Quick decision, Why this result, Technical evidence and Learn views; demo results remain operationally invalid and no engineering thresholds or causal explanations are fabricated.
- [x] Frontend regression tests cover custom playbook overrides, all reliability paths, reported-location handling, SHM threshold safety and the four-level result navigation; production build and lint gates pass.
