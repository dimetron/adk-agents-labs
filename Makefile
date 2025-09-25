export PATH=/opt/homebrew/bin:$PATH:.venv/bin

# [ gpt-oss:latest || mistral:latest ]
export DEFAULT_MODEL ?= gpt-oss:latest 
export OLLAMA_BASE_URL ?= http://localhost:11434

export UV_VENV_CLEAR=1

.PHONY: pull-model
pull-model:
	ollama pull $(DEFAULT_MODEL)

.PHONY: clean
clean:
	/bin/rm -rf .venv htmlcov .pytest_cache __pycache__ */__pycache__ */*/__pycache__
	ps -ef | grep adk | grep -v grep | tee | awk '{print $2}' | xargs -n1 -0 kill -9  || true

.PHONY: install
install: pull-model clean
	uv venv
	uv pip install google-adk
	uv pip install -r requirements.txt
	uv pip install pytest pytest-cov pytest-asyncio

.PHONY: test
test: install
	uv run python -m pytest || test $$? -eq 5

.PHONY: test-coverage
test-coverage: install
	uv run python -m pytest --cov=. --cov-report=term-missing --cov-report=html:htmlcov || test $$? -eq 5

web: install
	DEFAULT_MODEL=ollama/$(DEFAULT_MODEL) OLLAMA_BASE_URL=$(OLLAMA_BASE_URL) .venv/bin/adk web .