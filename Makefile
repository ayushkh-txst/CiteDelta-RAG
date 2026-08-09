.PHONY: help up down psql logs sync fmt lint types test check clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

up:  ## start postgres and wait for it to be healthy
	docker compose up -d --wait

down:  ## stop postgres (keeps data)
	docker compose down

nuke:  ## stop postgres and DELETE all data
	docker compose down -v

psql:  ## open a psql shell
	docker compose exec postgres psql -U citedelta -d citedelta

logs:  ## tail postgres logs
	docker compose logs -f postgres

sync:  ## install/refresh the environment
	uv sync --all-packages

fmt:  ## autoformat
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## lint without fixing
	uv run ruff check .
	uv run ruff format --check .

types:  ## strict type check
	uv run mypy

test:  ## run the test suite (against citedelta_test, never the dev db)
	uv run pytest

test-db-drop:  ## drop the test database (it is recreated on next run)
	docker compose exec postgres dropdb -U citedelta --if-exists citedelta_test

check: lint types test  ## everything CI runs