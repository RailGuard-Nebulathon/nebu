# RailGuard development rules

Mission: deliver a reproducible, leakage-safe implementation of all four NebulaX PS3 tracks.

- Never fabricate schemas, labels, metrics, results, or S-N constants.
- Never start expensive training automatically. Training must be an explicit CLI action.
- Inspect each source schema before making task-specific assumptions; never edit raw inputs.
- Raw, interim, processed, prediction, and checkpoint data remain out of version control.
- Preserve official identifiers and output spellings at submission boundaries.
- Split at file/case/group level, fit preprocessors inside folds, and test leakage guards.
- Keep common infrastructure in `railguard`; task-specific parsing and models stay isolated.
- Use typed, cohesive functions, `pathlib`, deterministic seeds, and standard logging.
- Gracefully degrade when optional deep-learning, explainability, or dashboard packages are absent.
- Every completed phase updates `docs/IMPLEMENTATION_STATUS.md` and passes focused tests.

Common commands: `pytest -q`, `ruff check .`, `python scripts/inspect_data.py --task all`,
`python scripts/build_manifests.py --task all`, `python scripts/train_deep.py --task door --dry-run`,
and `streamlit run dashboard/app.py`.

