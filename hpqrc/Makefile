.PHONY: help setup install test lint format clean train ablation figures

help:
	@echo "HPQRC - Available commands:"
	@echo "  make setup          - Install project in development mode"
	@echo "  make test           - Run test suite"
	@echo "  make lint           - Run linters (ruff, mypy)"
	@echo "  make format         - Format code (ruff, black)"
	@echo "  make train          - Run default training"
	@echo "  make ablation       - Run ablation experiments"
	@echo "  make figures        - Generate publication figures"
	@echo "  make clean          - Clean build artifacts"

setup: install

install:
	cd hpqrc && pip install -e ".[dev]"

test:
	cd hpqrc && pytest tests/ -v

test-cov:
	cd hpqrc && pytest tests/ --cov=src --cov-report=html --cov-report=term

lint:
	cd hpqrc && ruff check src/ tests/ scripts/
	cd hpqrc && mypy src/ --ignore-missing-imports

format:
	cd hpqrc && ruff check src/ tests/ scripts/ --fix
	cd hpqrc && black src/ tests/ scripts/

clean:
	cd hpqrc && rm -rf build/ dist/ *.egg-info
	cd hpqrc && rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	cd hpqrc && find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	cd hpqrc && find . -type f -name "*.pyc" -delete
	cd hpqrc && rm -rf outputs/

train:
	cd hpqrc && python scripts/train.py model=hpqrc data=acn_caltech

train-gpu:
	cd hpqrc && CUDA_VISIBLE_DEVICES=0 python scripts/train.py model=hpqrc data=acn_caltech hardware=t4

ablation:
	cd hpqrc && python scripts/run_ablation.py experiment=ablation_memory
	cd hpqrc && python scripts/run_ablation.py experiment=ablation_quantum

memory-benchmark:
	cd hpqrc && python scripts/run_memory_benchmark.py

figures:
	cd hpqrc && python scripts/generate_figures.py

evaluate:
	cd hpqrc && python scripts/evaluate.py

notebooks:
	cd hpqrc && jupyter notebook notebooks/

pre-commit:
	cd hpqrc && pre-commit install
	cd hpqrc && pre-commit run --all-files
