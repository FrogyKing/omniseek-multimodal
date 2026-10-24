.PHONY: install demo api frontend test lint build cost-report benchmark-dry-run gpu-smoke-test gpu-benchmark gpu-research evaluate

install:
	uv sync --locked
	npm --prefix frontend ci

demo:
	uv run python -m omniseek_multimodal.cli demo

api:
	uv run python -m omniseek_multimodal.cli serve

frontend:
	npm --prefix frontend run dev

test:
	uv run pytest

lint:
	uv run ruff check src modal_apps scripts tests
	uv run ruff format --check src modal_apps scripts tests
	uv run mypy
	npm --prefix frontend run lint
	npm --prefix frontend run format:check

build:
	npm --prefix frontend run build

cost-report:
	uv run python -m omniseek_multimodal.cli cost-report

benchmark-dry-run:
	uv run python -m omniseek_multimodal.cli benchmark-dry-run

gpu-smoke-test:
	uv run python -m omniseek_multimodal.cli gpu-smoke-test

gpu-benchmark:
	uv run python -m omniseek_multimodal.cli gpu-benchmark

gpu-research:
	uv run python -m omniseek_multimodal.cli gpu-research

evaluate:
	uv run python -m omniseek_multimodal.cli evaluate
