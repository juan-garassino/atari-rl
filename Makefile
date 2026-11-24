.PHONY: help install test smoke launch status videos clean

GAME ?=

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## uv sync with dev deps (CPU torch)
	uv sync --extra dev

test:  ## Run tests
	nice -n 10 uv run pytest -q

smoke:  ## 2-env Pong run, ~2k steps, niced
	nice -n 19 uv run atari-rl smoke

launch:  ## Start a RunPod training pod: make launch GAME=pong|breakout [ARGS="-f n_envs=32"]
	@if [ -z "$(GAME)" ]; then echo "Usage: make launch GAME=pong|breakout"; exit 1; fi
	gh workflow run deploy-runpod.yml -f game=$(GAME) $(ARGS)

status:  ## Live run state from MLflow: make status RUN=pong-20261124-180000
	@if [ -z "$(RUN)" ]; then echo "Usage: make status RUN=<run id>"; exit 1; fi
	@uv run atari-rl status $(RUN) 2>/dev/null

videos:  ## Download a run's eval replays to ~/Videos/rl-series/<run>: make videos RUN=pong-20261124-180000
	@if [ -z "$(RUN)" ]; then echo "Usage: make videos RUN=<run id>"; exit 1; fi
	@uv run atari-rl videos $(RUN)

clean:  ## Remove caches and local runs
	rm -rf .pytest_cache runs **/__pycache__
