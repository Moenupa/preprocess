.PHONY: style style-check quality test

check_dirs := src examples

# uvx with fallback, e.g.: 1. `uvx ruff check` 2. `ruff check`
TOOL := $(shell command -v uv >/dev/null 2>&1 && echo "uvx" || echo "")
RUN := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "")

all: style quality

# check for style, do nothing
style-check:
	$(TOOL) ruff check $(check_dirs)
	$(TOOL) ruff format --check $(check_dirs)

# check for style, and fix issues
style:
	$(TOOL) ruff check $(check_dirs) --fix
	$(TOOL) ruff format $(check_dirs)

# code quality checks
quality:
	$(TOOL) ty check $(check_dirs)

test:
	OPENAI_API_KEY=sk-test $(RUN) pytest tests