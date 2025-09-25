"""Native Ollama model driver that integrates directly with Google ADK."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field
from typing_extensions import override


logger = logging.getLogger("google_adk.ollama")


def _env_timeout() -> float:
    raw_timeout = os.getenv("OLLAMA_TIMEOUT", "120")
    try:
        return float(raw_timeout)
    except ValueError:
        logger.warning("Invalid OLLAMA_TIMEOUT=%s. Using 120 seconds.", raw_timeout)
        return 120.0


def _env_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _env_keep_alive() -> Optional[str]:
    value = os.getenv("OLLAMA_KEEP_ALIVE")
    return value if value else None


def _env_options() -> Dict[str, Any]:
    raw_options = os.getenv("OLLAMA_OPTIONS_JSON")
    if not raw_options:
        return {}

    try:
        parsed = json.loads(raw_options)
    except json.JSONDecodeError:
        logger.warning("Failed to parse OLLAMA_OPTIONS_JSON. Ignoring value.")
        return {}

    if not isinstance(parsed, dict):
        logger.warning(
            "OLLAMA_OPTIONS_JSON must be a JSON object. Got %s instead.",
            type(parsed).__name__,
        )
        return {}

    return parsed


def _schema_to_json(schema: types.Schema) -> Dict[str, Any]:
    schema_dict = schema.model_dump(exclude_none=True)

    type_value = schema_dict.get("type")
    if type_value is not None:
        if isinstance(type_value, types.Type):
            schema_dict["type"] = type_value.value.lower()
        else:
            schema_dict["type"] = str(type_value).lower()

    if "enum" in schema_dict:
        schema_dict["enum"] = [
            value.value if hasattr(value, "value") else value
            for value in schema_dict["enum"]
        ]

    if "properties" in schema_dict:
        new_props: Dict[str, Any] = {}
        for key, value in schema_dict["properties"].items():
            if isinstance(value, types.Schema):
                new_props[key] = _schema_to_json(value)
            else:
                new_props[key] = _schema_to_json(types.Schema.model_validate(value))
        schema_dict["properties"] = new_props

    if "items" in schema_dict:
        items = schema_dict["items"]
        if isinstance(items, types.Schema):
            schema_dict["items"] = _schema_to_json(items)
        else:
            schema_dict["items"] = _schema_to_json(
                types.Schema.model_validate(items)
            )

    return schema_dict


def _function_declaration_to_tool(
    function_declaration: types.FunctionDeclaration,
) -> Dict[str, Any]:
    assert function_declaration.name

    parameters: Dict[str, Any] = {"type": "object", "properties": {}}
    if function_declaration.parameters:
        parameters = _schema_to_json(function_declaration.parameters)
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        if function_declaration.parameters.required:
            parameters["required"] = function_declaration.parameters.required

    tool: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": function_declaration.name,
            "description": function_declaration.description or "",
            "parameters": parameters,
        },
    }

    return tool


def _extract_function_tools(llm_request: LlmRequest) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    config = llm_request.config
    if not config or not config.tools:
        return tools

    for tool in config.tools:
        if not isinstance(tool, types.Tool):
            continue
        if not tool.function_declarations:
            continue
        for declaration in tool.function_declarations:
            tools.append(_function_declaration_to_tool(declaration))

    return tools


class OllamaLlm(BaseLlm):
    """Minimal native Ollama driver.

    This driver translates Google ADK requests into Ollama's `/api/chat` calls
    and converts responses back into `LlmResponse` objects.
    """

    base_url: str = Field(default_factory=_env_base_url)
    timeout: float = Field(default_factory=_env_timeout)
    verify: bool = True
    headers: Dict[str, str] = Field(default_factory=dict)
    default_options: Dict[str, Any] = Field(default_factory=_env_options)
    keep_alive: Optional[str] = Field(default_factory=_env_keep_alive)

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        verify: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        default_options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> None:
        resolved_base_url = (base_url or _env_base_url()).rstrip("/")
        resolved_timeout = timeout if timeout is not None else _env_timeout()
        resolved_verify = True if verify is None else verify
        resolved_headers = dict(headers or {})

        if "Accept" not in resolved_headers:
            resolved_headers["Accept"] = "application/json"

        api_key = os.getenv("OLLAMA_API_KEY")
        if api_key:
            resolved_headers.setdefault("Authorization", f"Bearer {api_key}")

        merged_options = dict(_env_options())
        if default_options:
            merged_options.update(default_options)

        resolved_keep_alive = (
            keep_alive if keep_alive is not None else _env_keep_alive()
        )

        super().__init__(
            model=model,
            base_url=resolved_base_url,
            timeout=resolved_timeout,
            verify=resolved_verify,
            headers=resolved_headers,
            default_options=merged_options,
            keep_alive=resolved_keep_alive,
        )

    @classmethod
    def from_env(cls, model: str) -> "OllamaLlm":
        """Helper for creating a driver with environment-derived defaults."""

        return cls(model=model)

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        return model.split("/", 1)[1] if model.startswith("ollama/") else model

    @staticmethod
    def _map_role(role: Optional[str]) -> str:
        if role in ("model", "assistant"):
            return "assistant"
        if role == "system":
            return "system"
        if role == "tool":
            return "tool"
        return "user"

    @staticmethod
    def _part_to_text(part: types.Part) -> Optional[str]:
        if part.text:
            return part.text
        if part.function_call:
            payload = {
                "function_call": {
                    "name": part.function_call.name,
                    "args": part.function_call.args,
                }
            }
            return json.dumps(payload)
        if part.function_response:
            return json.dumps(part.function_response.response)
        if part.executable_code:
            return part.executable_code.code
        if part.code_execution_result:
            return part.code_execution_result.output
        if part.inline_data and part.inline_data.data:
            return "[binary data omitted]"
        return None

    def _convert_messages(self, llm_request: LlmRequest) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []

        if llm_request.config and llm_request.config.system_instruction:
            messages.append(
                {
                    "role": "system",
                    "content": llm_request.config.system_instruction,
                }
            )

        for content in llm_request.contents or []:
            text_fragments: List[str] = []
            for part in content.parts or []:
                part_text = self._part_to_text(part)
                if part_text:
                    text_fragments.append(part_text)

            if not text_fragments:
                continue

            messages.append(
                {
                    "role": self._map_role(content.role),
                    "content": "\n".join(text_fragments),
                }
            )

        return messages

    def _build_options(self, llm_request: LlmRequest) -> Dict[str, Any]:
        options = dict(self.default_options)
        cfg = llm_request.config

        if cfg.temperature is not None:
            options["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            options["top_p"] = cfg.top_p
        if cfg.top_k is not None:
            options["top_k"] = cfg.top_k
        if cfg.max_output_tokens is not None:
            options["num_predict"] = cfg.max_output_tokens
        if cfg.stop_sequences:
            options["stop"] = cfg.stop_sequences

        return options

    def _build_payload(self, llm_request: LlmRequest, stream: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self._normalize_model_name(llm_request.model or self.model),
            "messages": self._convert_messages(llm_request),
            "stream": stream,
        }

        options = self._build_options(llm_request)
        if options:
            payload["options"] = options

        tools = _extract_function_tools(llm_request)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if (
            llm_request.config.response_schema
            or llm_request.config.response_mime_type == "application/json"
        ):
            payload["format"] = "json"

        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive

        return payload

    def _extract_usage(self, data: Dict[str, Any]) -> Optional[types.GenerateContentResponseUsageMetadata]:
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")

        if prompt_tokens is None and completion_tokens is None:
            return None

        return types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens or 0,
            candidates_token_count=completion_tokens or 0,
            total_token_count=(prompt_tokens or 0) + (completion_tokens or 0),
        )

    @staticmethod
    def _map_finish_reason(reason: Optional[str]) -> Optional[types.FinishReason]:
        if not reason:
            return None
        normalized = reason.lower()
        if normalized in {"stop", "stopped"}:
            return types.FinishReason.STOP
        if normalized in {"length", "max_tokens"}:
            return types.FinishReason.MAX_TOKENS
        return types.FinishReason.FINISH_REASON_UNSPECIFIED

    def _build_llm_response(self, data: Dict[str, Any]) -> LlmResponse:
        if "error" in data:
            raise RuntimeError(f"Ollama error: {data['error']}")

        message = data.get("message") or {}
        parts: List[types.Part] = []

        content_text = message.get("content", "")

        # Check if content_text contains JSON function calls that should be parsed
        # some models have broken response format
        function_calls_parsed = False
        self.re_parse_tools(content_text, function_calls_parsed, parts)

        # Also check for proper tool_calls in the message structure
        for tool_call in message.get("tool_calls", []) or []:
            function_data = tool_call.get("function", {})
            name = function_data.get("name")
            arguments = function_data.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.debug("Failed to parse tool arguments: %s", arguments)
            if not isinstance(arguments, dict):
                arguments = {"__raw": arguments}
            part = types.Part.from_function_call(name=name, args=arguments)
            part.function_call.id = tool_call.get("id")
            parts.append(part)

        content = None
        if parts:
            content = types.Content(role="model", parts=parts)

        return LlmResponse(
            content=content,
            usage_metadata=self._extract_usage(data),
            finish_reason=self._map_finish_reason(data.get("done_reason")),
        )

    def re_parse_tools(self, content_text, function_calls_parsed, parts):
        if content_text:
            # Look for JSON function calls in the text content
            json_pattern = r'\[?\{[^}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}\]?'
            matches = re.findall(json_pattern, content_text)

            if matches:
                # Parse JSON function calls from text
                for match in matches:
                    try:
                        # Clean up the match - remove surrounding brackets if present
                        clean_match = match.strip('[]')
                        func_data = json.loads(clean_match)

                        name = func_data.get("name")
                        arguments = func_data.get("arguments", {})

                        if name:
                            part = types.Part.from_function_call(name=name, args=arguments)
                            parts.append(part)
                            function_calls_parsed = True

                    except json.JSONDecodeError:
                        logger.debug("Failed to parse function call from text: %s", match)
                        continue

                # If we found function calls in text, remove them from the text content
                if function_calls_parsed:
                    cleaned_text = re.sub(json_pattern, '', content_text).strip()
                    # Only add text part if there's meaningful content left
                    if cleaned_text and not re.match(r'^\s*$', cleaned_text):
                        parts.insert(0, types.Part.from_text(text=cleaned_text))
                else:
                    # No function calls found, add the original text
                    parts.append(types.Part.from_text(text=content_text))
            else:
                # No function call patterns found, add the original text
                parts.append(types.Part.from_text(text=content_text))

    async def _stream_chat(self, client: httpx.AsyncClient, payload: Dict[str, Any]) -> AsyncGenerator[LlmResponse, None]:
        url = "/api/chat"
        aggregated_text = ""
        usage: Optional[types.GenerateContentResponseUsageMetadata] = None
        finish_reason: Optional[str] = None
        tool_calls: Dict[int, Dict[str, Any]] = {}

        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            buffer = ""
            async for chunk in response.aiter_text():
                if not chunk:
                    continue
                buffer += chunk

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)
                    if "error" in data:
                        raise RuntimeError(f"Ollama error: {data['error']}")

                    if data.get("message") and data["message"].get("content"):
                        piece = data["message"]["content"]
                        aggregated_text += piece
                        yield LlmResponse(
                            content=types.Content(
                                role="model",
                                parts=[types.Part.from_text(text=piece)],
                            ),
                            partial=True,
                        )

                    if data.get("message") and data["message"].get("tool_calls"):
                        for index, call in enumerate(data["message"]["tool_calls"]):
                            entry = tool_calls.setdefault(
                                index,
                                {
                                    "id": call.get("id"),
                                    "name": "",
                                    "arguments": "",
                                },
                            )
                            if call.get("id"):
                                entry["id"] = call["id"]
                            function = call.get("function", {})
                            if function.get("name"):
                                entry["name"] = function.get("name")
                            args_fragment = function.get("arguments", "")
                            if args_fragment:
                                entry["arguments"] += args_fragment

                    if data.get("done"):
                        usage = self._extract_usage(data)
                        finish_reason = data.get("done_reason")

        parts: List[types.Part] = []
        if aggregated_text:
            parts.append(types.Part.from_text(text=aggregated_text))

        if tool_calls:
            for index in sorted(tool_calls.keys()):
                call_info = tool_calls[index]
                arguments: Any = call_info["arguments"]
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.debug("Failed to parse streamed tool args: %s", arguments)
                if not isinstance(arguments, dict):
                    arguments = {"__raw": arguments}
                part = types.Part.from_function_call(
                    name=call_info["name"], args=arguments
                )
                part.function_call.id = call_info.get("id")
                parts.append(part)

        final_content = None
        if parts:
            final_content = types.Content(role="model", parts=parts)

        yield LlmResponse(
            content=final_content,
            usage_metadata=usage,
            finish_reason=self._map_finish_reason(finish_reason),
        )

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self._maybe_append_user_content(llm_request)

        payload = self._build_payload(llm_request, stream)

        client_kwargs = {
            "base_url": self.base_url,
            "timeout": httpx.Timeout(self.timeout),
            "headers": self.headers,
            "verify": self.verify,
        }

        async with httpx.AsyncClient(**client_kwargs) as client:
            if stream:
                async for response in self._stream_chat(client, payload):
                    yield response
            else:
                http_response = await client.post("/api/chat", json=payload)
                http_response.raise_for_status()
                data = http_response.json()
                yield self._build_llm_response(data)

    @classmethod
    @override
    def supported_models(cls) -> List[str]:
        return [r"ollama/.*"]


try:
    from google.adk.models.registry import LLMRegistry

    LLMRegistry.register(OllamaLlm)
except Exception as exc:  # pragma: no cover - defensive registration
    logger.debug("Unable to register OllamaLlm with LLMRegistry: %s", exc)


