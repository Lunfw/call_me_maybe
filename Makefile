PY	= python3
RED	= \033[1;31m
GREEN	= \033[1;32m
CYAN	= \033[1;36m
RESET	= \033[0m

all: build
	@echo "\nUsage:"; \
	echo "$(CYAN)make run$(RESET) - run the program, followed by a prompt"; \
	echo "$(CYAN)make debug$(RESET) - run the program in debug mode, followed by a prompt"; \
	echo "$(CYAN)make clean$(RESET) - remove **/__pycache__, .venv and .pyc files"; \
	echo "$(CYAN)make lint$(RESET) - run flake8 & mypy"; \
	echo "$(CYAN)make lint-strict$(RESET) - run flake8 & mypy (strict mode)"

build:
	@if [ ! -d .venv ]; then \
		echo "\n$(CYAN)Creating a virtual environment...$(RESET)"; \
		$(PY) -m venv .venv; \
		echo "$(GREEN)A virtual environment was created under $(RED).venv.$(RESET)"; \
		echo "You can use the following commands:"; \
		echo "$(CYAN)source .venv/bin/activate$(RESET) - activate the venv"; \
		echo "$(CYAN)source .venv/bin/deactivate$(RESET) - deactivate the venv"; \
		.venv/bin/pip install -r requirements.txt -qqq; \
	else \
		echo "\n$(RED)A virtual environment already exists.$(RESET)"; \
		echo "You can use the following commands:"; \
		echo "$(CYAN)source .venv/bin/activate$(RESET) - activate the venv"; \
		echo "$(CYAN)source .venv/bin/deactivate$(RESET) - deactivate the venv"; \
	fi

run:
	@if [ ! -d .venv ]; then \
		make build; \
	fi
	.venv/bin/uv run $(PY) -m src

lint:
	@if [ ! -d .venv ]; then \
		make build; \
	fi
	@echo "\0"
	.venv/bin/flake8 src
	.venv/bin/mypy src

lint-strict:
	@if [ ! -d .venv ]; then \
		make build; \
	fi
	@echo "\0"
	.venv/bin/flake8 src --select=F
	.venv/bin/mypy src --strict

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -exec rm -f {} +
	rm -rf .venv

.PHONY: build clean lint lint-strict
