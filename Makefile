export UV_CACHE_DIR		= .cache/uv
export HF_HOME			= .cache/hugging/face
export TRANSFORMERS_CACHE	= .cache/hugging
export PIP_CACHE_DIR		= .cache/pip

PY	= python3
VENV	= .venv
UV	= uv
PDB	= pudb
RED	= \033[1;31m
GREEN	= \033[1;32m
CYAN	= \033[1;36m
RESET	= \033[0m
ARGS	= $(filter-out $@, $(MAKECMDGOALS))

all: build
	@echo "\nUsage:"; \
	echo "$(CYAN)make run$(RESET) - run the program, followed by a prompt"; \
	echo "$(CYAN)make debug$(RESET) - run the program in debug mode, followed by a prompt"; \
	echo "$(CYAN)make clean$(RESET) - remove **/__pycache__, $(VENV) and .pyc files"; \
	echo "$(CYAN)make lint$(RESET) - run flake8 & mypy"; \
	echo "$(CYAN)make lint-strict$(RESET) - run flake8 & mypy (strict mode)"

build:
	@if [ ! -d .venv ]; then \
		echo "\n$(CYAN)Syncing dependencies...$(RESET)"; \
		$(UV) sync; \
		echo "$(GREEN)A virtual environment was created under $(RED)$(VENV).$(RESET)"; \
		echo "You can use the following commands:"; \
		echo "$(CYAN)source .venv/bin/activate$(RESET) - activate the venv"; \
		echo "$(CYAN)source .venv/bin/deactivate$(RESET) - deactivate the venv"; \
	fi

run: build
	$(UV) run $(PY) -m src $(ARGS)

%:
	@:


debug: build
	$(VENV)/bin/$(PDB) -m src $(ARGS)

lint: build
	@echo "\0"; \
	$(UV) run flake8 src; \
	$(UV) run mypy src


lint-strict: build
	@echo "\0"; \
	$(UV) run $(VENV)/bin/flake8 src --select=F; \
	$(UV) run $(VENV)/bin/mypy src --strict

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -exec rm -f {} +
	find . -name ".mypy_cache" -exec rm -rf {} +
	rm -rf .venv
	rm -rf .cache
	rm -rf data/output/*.json

.PHONY: all build clean lint lint-strict
