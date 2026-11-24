.PHONY: help install test smoke clean

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## uv sync with dev deps (CPU torch)
	uv sync --extra dev

test:  ## Run tests
	nice -n 10 uv run pytest -q

smoke:  ## 2-env Pong run, ~2k steps, niced
	nice -n 19 uv run atari-rl smoke

clean:  ## Remove caches and local runs
	rm -rf .pytest_cache runs **/__pycache__
