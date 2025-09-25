export PATH=$PATH:.venv/bin
export DEFAULT_MODEL ?= mistral:latest #gpt-oss:latest
export OLLAMA_BASE_URL ?= http://localhost:11434

.PHONY: pull-model
pull-model:
	ollama pull $(DEFAULT_MODEL)

.PHONY: install
install: pull-model
install:
	uv venv --clear
	uv pip install google-adk
	uv pip install -r requirements.txt

web: install
	killall uv || true
	DEFAULT_MODEL=ollama/$(DEFAULT_MODEL) OLLAMA_BASE_URL=$(OLLAMA_BASE_URL) .venv/bin/adk web .
	