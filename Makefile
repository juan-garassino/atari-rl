.PHONY: help install test smoke smoke-scratch cartpole launch status videos play clean

GAME ?=

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## uv sync with dev deps (CPU torch)
	uv sync --extra dev

test:  ## Run tests
	nice -n 10 uv run pytest -q

smoke:  ## 2-env Pong run, ~2k steps, niced
	nice -n 19 uv run atari-rl smoke

smoke-scratch:  ## Same smoke run with the from-scratch PPO
	ATARIRL_ALGO=scratch nice -n 19 uv run atari-rl smoke

cartpole:  ## Scratch PPO sanity check on CartPole-v1 (should reach ~500), niced
	nice -n 19 uv run atari-rl cartpole

launch:  ## Start a RunPod training pod: make launch GAME=pong|breakout [ARGS="-f n_envs=32"]
	@if [ -z "$(GAME)" ]; then echo "Usage: make launch GAME=pong|breakout"; exit 1; fi
	gh workflow run deploy-runpod.yml -f game=$(GAME) $(ARGS)

status:  ## Live run state from MLflow: make status RUN=pong-20261124-180000
	@if [ -z "$(RUN)" ]; then echo "Usage: make status RUN=<run id>"; exit 1; fi
	@uv run atari-rl status $(RUN) 2>/dev/null

videos:  ## Download eval replays + best_model.zip to ~/Videos/rl-series/<run>: make videos RUN=pong-20261124-180000
	@if [ -z "$(RUN)" ]; then echo "Usage: make videos RUN=<run id>"; exit 1; fi
	@uv run atari-rl videos $(RUN)

play:  ## Replay a run's best model locally, full game → mp4: make play RUN=pong-... GAME=pong
	@if [ -z "$(RUN)" ] || [ -z "$(GAME)" ]; then echo "Usage: make play RUN=<run id> GAME=pong|breakout"; exit 1; fi
	nice -n 19 uv run atari-rl eval --game $(GAME) --model $(HOME)/Videos/rl-series/$(RUN)/best_model.zip \
		--episodes 1 --video $(HOME)/Videos/rl-series/$(RUN)/local_play.mp4

clean:  ## Remove caches and local runs
	rm -rf .pytest_cache runs **/__pycache__
