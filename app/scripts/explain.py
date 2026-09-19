"""Generate model-evidence artifacts from a trusted local bundle."""

import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("door", "acv", "corrugation", "shm"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    from railguard.explainability.feature_importance import feature_importance
    from railguard.models.serialization import load_classical_bundle

    model, _ = load_classical_bundle(args.checkpoint)
    print(feature_importance(model, model.feature_names_))


if __name__ == "__main__":
    main()
