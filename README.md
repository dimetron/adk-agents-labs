# ADK Agents Labs

This repository showcases a set of Google ADK (Agent Development Kit) experiments that connect locally running Ollama large language models with multi-agent workflows. The labs demonstrate two complementary experiences:

- Coordinated travel-planning agents that route users between brainstorming and detailed itinerary planning.
- A collaborative writers room that researches, drafts, critiques, and persists historical biopic movie pitches.


## Prerequisites

- Python 3.10+
- `uv` for virtual environment management and dependency installation
- [Ollama](https://ollama.com/) running locally with the `gpt-oss` model available
- (Optional) A `.env` file to override defaults such as `DEFAULT_MODEL`, `MODEL`, `OLLAMA_BASE_URL`, or other Ollama settings consumed by `ollama_llm.py`


## Quick Start

```bash
make web
```

The `web` target performs the following:

1. Pulls the `gpt-oss` model from Ollama (target `pull-model`).
2. Creates a fresh virtual environment with `uv` and installs `google-adk` plus dependencies from `requirements.txt` (target `install`).
3. Launches the ADK web interface using the default model and base URL exported in the Makefile.

After the web server starts, visit the printed URL to interact with the agents.


## Project Structure

- `parent-agents/agent.py` – Defines a root "steering" agent that triages users between `travel_brainstormer` and `attractions_planner`. The agents share state via ADK tools to accumulate attraction selections.
- `workflow-agents/agent.py` – Assembles a multi-step writing workflow. Agents collaborate to research a historical figure, iteratively refine a plot outline, critique the result, and persist a pitch file to the `movie_pitches/` directory.
- `ollama_llm.py` – Provides a native Ollama driver that adapts ADK requests to Ollama's chat API, handling streaming, function calling, and environment-driven configuration.
- `callback_logging.py` – Hooks ADK callbacks to log user prompts, model responses, and tool calls for traceability.
- `movie_pitches/` – Destination folder for generated movie pitch files.
- `Makefile` – Encapsulates setup and web launch commands using `uv` and Ollama defaults.
- `requirements.txt` – Additional Python dependencies (LangChain, Wikipedia tooling, dotenv, LiteLLM, httpx).


## Configuring Models

Environment variables control which model is used and how the driver connects to Ollama:

- `MODEL` – Highest priority. Accepts native ADK model names or `ollama/<model-name>` to force the Ollama driver.
- `DEFAULT_MODEL` – Used when `MODEL` is unset. Defaults to `ollama/gpt-oss:latest` via the Makefile.
- `OLLAMA_BASE_URL`, `OLLAMA_TIMEOUT`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_OPTIONS_JSON`, `OLLAMA_API_KEY` – Passed into `OllamaLlm` for connectivity, timeouts, and advanced options.

The driver normalizes responses, exposes token usage metadata, and streams partial outputs when requested by ADK.


## Extending the Labs

- Add new ADK agents or tools alongside existing ones to explore different coordination patterns.
- Update `workflow-agents/agent.py` to introduce additional review cycles or output formats.
- Modify `parent-agents/agent.py` to branch into other specialized travel planners.
- Customize the `write_file` tool to persist outputs in alternative formats (Markdown, JSON) or storage backends.

When introducing new dependencies, remember to add them to `requirements.txt` and rerun `make install` or `make web`.


## Troubleshooting

- Ensure Ollama is running and the target model is available. Use `ollama pull gpt-oss:latest` if the Makefile step fails.
- If the ADK web server cannot connect to Ollama, verify `OLLAMA_BASE_URL` and network access.
- For verbose logging, configure Python's logging level (e.g., via `LOGLEVEL=INFO make web`).
- On macOS, `killall uv` in the `web` target stops any stale ADK process before relaunching.


## License

No license information is provided. Supply one if you intend to distribute or open-source the project.


