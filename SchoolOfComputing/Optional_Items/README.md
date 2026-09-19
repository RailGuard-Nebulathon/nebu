# RailGuard AI optional submission materials

This directory contains the optional NebulaX PS3 methodology write-up and
task-specific development artifacts. It excludes the supplied raw datasets,
test predictions, and compulsory application deliverables.

## Contents

- `write_up.md`: combined methodology, feature engineering, model selection,
  evaluation, assumptions, and reproducibility write-up.
- Each subsystem's `code` directory: relevant source and configurations.
- Each subsystem's `model` directory: its saved competition bundle.
- `MANIFEST.sha256`: SHA-256 checksums for package verification.

Saved bundles are present for all four subsystems. The Door source artifacts
arrived with shifted filenames; `Door/model` normalizes them and documents the
exact mapping without altering their content.

Serialized Joblib files must only be loaded from this trusted local package.

