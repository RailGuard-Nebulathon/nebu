"""List generated locations; removal requires an explicit user command outside this helper."""

from pathlib import Path


def main() -> None:
    paths = [Path("data/interim"), Path("data/processed"), Path("data/manifests"), Path("outputs")]
    print("Generated locations (not removed automatically):")
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()

