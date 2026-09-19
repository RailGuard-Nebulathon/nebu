.PHONY: install install-dev test lint format inspect manifest dashboard api web web-install train-competition smoke clean
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
api:
	uvicorn railguard_api.main:app --app-dir web/backend --reload
web-install:
	npm --prefix web/frontend install
web:
	npm --prefix web/frontend run dev
train-competition:
	python scripts/train_competition.py --task all --raw-root "$(RAW_ROOT)"
smoke:
	pytest -q tests/test_end_to_end_synthetic.py
clean:
	python scripts/clean_generated.py
