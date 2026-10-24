.PHONY: install lint format typecheck test test-unit evals evals-live run doctor docker compose-up compose-models

PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

install:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -e ".[dev]"

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

typecheck:
	$(BIN)/mypy

test:            ## everything, including Postgres/Redis/Docker integration tests when reachable
	$(BIN)/pytest -q -rs --cov

test-unit:       ## no external services
	$(BIN)/pytest -q -m "not integration"

evals:           ## deterministic harness check (no GPUs)
	$(BIN)/sentientai eval run --scripted

evals-live:      ## against the configured model servers
	$(BIN)/sentientai -c sentient.yaml eval run --output reports/evals.json

run:
	$(BIN)/sentientai -c sentient.yaml run

doctor:
	$(BIN)/sentientai -c sentient.yaml doctor

docker:
	docker build -t sentientai:latest .

compose-up:
	docker compose -f deploy/docker-compose.yml up -d

compose-models:
	docker compose -f deploy/docker-compose.yml --profile models up -d
