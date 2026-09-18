.PHONY: install install-dev test lint format inspect manifest dashboard smoke clean
install:
	pip install -e .
install-dev:
	pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check .
format:
	ruff format .
inspect:
	python scripts/inspect_data.py --task all
manifest:
	python scripts/build_manifests.py --task all
sequences:
	@echo "Run scripts/build_sequences.py per task with --raw-root and --output"
dashboard:
	streamlit run dashboard/app.py
smoke:
	pytest -q tests/test_end_to_end_synthetic.py
clean:
	python scripts/clean_generated.py
