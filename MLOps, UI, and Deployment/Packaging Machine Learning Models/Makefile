SHELL := /bin/bash

.PHONY: sync train figures notebooks notebooks-exec notebooks-all test api cli-info verify docker-build docker-run

sync:
	UV_CACHE_DIR=.uv-cache uv sync

train:
	MPLCONFIGDIR=.mplconfig UV_CACHE_DIR=.uv-cache uv run --no-sync python scripts/train_model.py

figures:
	MPLCONFIGDIR=.mplconfig UV_CACHE_DIR=.uv-cache uv run --no-sync python scripts/generate_figures.py

notebooks:
	UV_CACHE_DIR=.uv-cache uv run --no-sync python scripts/generate_notebooks.py

notebooks-exec:
	MPLCONFIGDIR=.mplconfig UV_CACHE_DIR=.uv-cache uv run --no-sync python scripts/execute_notebooks.py --notebook-dir notebooks

notebooks-all: notebooks notebooks-exec

test:
	MPLCONFIGDIR=.mplconfig UV_CACHE_DIR=.uv-cache uv run --no-sync pytest -q

api:
	MPLCONFIGDIR=.mplconfig UV_CACHE_DIR=.uv-cache uv run --no-sync uvicorn api.main:app --host 0.0.0.0 --port 8000

cli-info:
	UV_CACHE_DIR=.uv-cache uv run --no-sync python -m ml_package.cli.predict --model-path models/iris_model.pkl info

verify:
	MPLCONFIGDIR=.mplconfig UV_CACHE_DIR=.uv-cache uv run --no-sync python scripts/verify_project.py

docker-build:
	docker build -t ml-packaging:latest .

docker-run:
	docker run --rm -p 8000:8000 ml-packaging:latest
