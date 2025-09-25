import os
import sys
import logging

from typing import List, Union

from dotenv import load_dotenv

from google.adk import Agent
from ollama_llm import OllamaLlm
from google.adk.tools.tool_context import ToolContext
from google.genai import types

sys.path.append("..")
from callback_logging import log_query_to_model, log_model_response


load_dotenv()


ModelSpec = Union[str, OllamaLlm]


def _ollama_model(model_value: str) -> OllamaLlm:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    logging.info(
        "Using Ollama model '%s' via native driver (base %s)", model_value, base_url
    )
    return OllamaLlm(model=model_value, base_url=base_url)


def _resolve_model() -> ModelSpec:
    env_model = os.getenv("MODEL")
    if env_model:
        if env_model.startswith("ollama/"):
            return _ollama_model(env_model)
        return env_model

    default_model = os.getenv("DEFAULT_MODEL")
    if default_model and default_model.startswith("ollama/"):
        return _ollama_model(default_model)

    fallback_model = default_model or "ollama/gpt-oss:latest"
    if fallback_model.startswith("ollama/"):
        return _ollama_model(fallback_model)
    return fallback_model


def save_attractions_to_state(
    tool_context: ToolContext,
    attractions: List[str],
) -> dict[str, str]:
    """Saves the list of attractions to state["attractions"]."""
    existing_attractions = tool_context.state.get("attractions", [])
    tool_context.state["attractions"] = existing_attractions + attractions
    return {"status": "success"}


model_spec = _resolve_model()


attractions_planner = Agent(
    name="attractions_planner",
    model=model_spec,
    description="Build a list of attractions to visit in a country.",
    instruction="""
        - Provide the user options for attractions to visit within their selected country.
        - When they reply, use your tool to save their selected attraction and then provide more possible attractions.
        - If they ask to view the list, provide a bulleted list of { attractions? } and then suggest some more.
        """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    tools=[save_attractions_to_state],
)


travel_brainstormer = Agent(
    name="travel_brainstormer",
    model=model_spec,
    description="Help a user decide what country to visit.",
    instruction="""
        Provide a few suggestions of popular countries for travelers.

        Help a user identify their primary goals of travel:
        adventure, leisure, learning, shopping, or viewing art

        Identify countries that would make great destinations
        based on their priorities.
        """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
)


root_agent = Agent(
    name="steering",
    model=model_spec,
    description="Start a user on a travel adventure.",
    instruction="""
        Ask the user if they know where they'd like to travel or if they need some help deciding.
        If they need help deciding, send them to 'travel_brainstormer'. If they know what country they'd like to visit, send them to the 'attractions_planner'.
        """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
    ),
    sub_agents=[travel_brainstormer, attractions_planner],
)
