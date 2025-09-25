"""Unit tests for OllamaLlm class."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from ollama_llm import OllamaLlm


class TestOllamaLlm:
    """Test cases for OllamaLlm class."""

    def test_init_with_defaults(self):
        """Test OllamaLlm initialization with default values."""
        llm = OllamaLlm(model="test-model")
        assert llm.model == "test-model"
        assert llm.base_url == "http://localhost:11434"
        assert llm.timeout == 120.0
        assert llm.keep_alive is None
        assert llm.default_options == {}

    def test_init_with_custom_values(self):
        """Test OllamaLlm initialization with custom values."""
        llm = OllamaLlm(
            model="custom-model",
            base_url="http://custom:8080",
            timeout=60.0,
            keep_alive="5m"
        )
        assert llm.model == "custom-model"
        assert llm.base_url == "http://custom:8080"
        assert llm.timeout == 60.0
        assert llm.keep_alive == "5m"

    def test_normalize_model_name(self):
        """Test model name normalization."""
        # Test the static method directly
        assert OllamaLlm._normalize_model_name("ollama/mistral:latest") == "mistral:latest"
        assert OllamaLlm._normalize_model_name("mistral:latest") == "mistral:latest"
        assert OllamaLlm._normalize_model_name("mistral") == "mistral"
        assert OllamaLlm._normalize_model_name("ollama/llama2") == "llama2"

    def test_part_to_text_static_method(self):
        """Test the _part_to_text static method."""
        # Mock a part with text
        mock_part = MagicMock()
        mock_part.text = "Hello, world!"
        mock_part.function_call = None
        mock_part.function_response = None
        mock_part.executable_code = None
        mock_part.code_execution_result = None
        mock_part.inline_data = None
        
        result = OllamaLlm._part_to_text(mock_part)
        assert result == "Hello, world!"
        
        # Mock a part without any content
        mock_part_empty = MagicMock()
        mock_part_empty.text = None
        mock_part_empty.function_call = None
        mock_part_empty.function_response = None
        mock_part_empty.executable_code = None
        mock_part_empty.code_execution_result = None
        mock_part_empty.inline_data = None
        
        result = OllamaLlm._part_to_text(mock_part_empty)
        assert result is None

    def test_map_role_static_method(self):
        """Test the _map_role static method."""
        assert OllamaLlm._map_role("user") == "user"
        assert OllamaLlm._map_role("model") == "assistant"
        assert OllamaLlm._map_role("assistant") == "assistant"
        assert OllamaLlm._map_role(None) == "user"
        assert OllamaLlm._map_role("unknown") == "user"

    def test_map_finish_reason_static_method(self):
        """Test the _map_finish_reason static method."""
        from google.genai import types
        assert OllamaLlm._map_finish_reason("stop") == types.FinishReason.STOP
        assert OllamaLlm._map_finish_reason("length") == types.FinishReason.MAX_TOKENS
        assert OllamaLlm._map_finish_reason(None) is None

    def test_extract_usage_method(self):
        """Test the _extract_usage method."""
        llm = OllamaLlm(model="test-model")
        
        # Test with data containing usage info
        data_with_usage = {
            "prompt_eval_count": 10,
            "eval_count": 20,
            "total_duration": 1000000000,
            "prompt_eval_duration": 500000000,
            "eval_duration": 400000000
        }
        
        usage = llm._extract_usage(data_with_usage)
        assert usage is not None
        
        # Test with empty data
        usage_empty = llm._extract_usage({})
        assert usage_empty is None

    @pytest.mark.asyncio
    async def test_generate_content_http_error(self):
        """Test content generation with HTTP error."""
        llm = OllamaLlm(model="test-model")
        
        with patch('ollama_llm.httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Simulate HTTP error
            mock_client.post.side_effect = httpx.HTTPError("Connection failed")
            
            mock_request = MagicMock()
            mock_request.contents = []
            mock_request.model = "test-model"
            mock_request.stream = False
            
            with pytest.raises(Exception):
                await llm.generate_content_async(mock_request)

    def test_from_env_class_method(self):
        """Test the from_env class method."""
        with patch.dict('os.environ', {
            'OLLAMA_BASE_URL': 'http://test:8080',
            'OLLAMA_TIMEOUT': '60'
        }):
            llm = OllamaLlm.from_env("test-model")
            assert llm.model == "test-model"
            assert llm.base_url == "http://test:8080"
            assert llm.timeout == 60.0

    def test_supported_models_class_method(self):
        """Test the supported_models class method."""
        models = OllamaLlm.supported_models()
        assert isinstance(models, list)
        # Should return some common model names
        assert len(models) > 0

    def test_env_timeout_valid(self):
        """Test environment timeout parsing with valid value."""
        with patch.dict('os.environ', {'OLLAMA_TIMEOUT': '60'}):
            from ollama_llm import _env_timeout
            assert _env_timeout() == 60.0

    def test_env_timeout_invalid(self):
        """Test environment timeout parsing with invalid value."""
        with patch.dict('os.environ', {'OLLAMA_TIMEOUT': 'invalid'}):
            from ollama_llm import _env_timeout
            assert _env_timeout() == 120.0  # default

    def test_env_base_url(self):
        """Test environment base URL parsing."""
        with patch.dict('os.environ', {'OLLAMA_BASE_URL': 'http://custom:8080/'}):
            from ollama_llm import _env_base_url
            assert _env_base_url() == "http://custom:8080"

    def test_env_keep_alive(self):
        """Test environment keep alive parsing."""
        with patch.dict('os.environ', {'OLLAMA_KEEP_ALIVE': '5m'}):
            from ollama_llm import _env_keep_alive
            assert _env_keep_alive() == "5m"

    def test_env_options_valid(self):
        """Test environment options parsing with valid JSON."""
        options_json = '{"temperature": 0.7, "num_predict": 100}'
        with patch.dict('os.environ', {'OLLAMA_OPTIONS_JSON': options_json}):
            from ollama_llm import _env_options
            options = _env_options()
            assert options["temperature"] == 0.7
            assert options["num_predict"] == 100

    def test_env_options_invalid(self):
        """Test environment options parsing with invalid JSON."""
        with patch.dict('os.environ', {'OLLAMA_OPTIONS_JSON': 'invalid-json'}):
            from ollama_llm import _env_options
            assert _env_options() == {}

    def test_build_options_method(self):
        """Test the _build_options method."""
        llm = OllamaLlm(model="test-model")
        mock_request = MagicMock()
        mock_request.generation_config = None
        
        options = llm._build_options(mock_request)
        assert isinstance(options, dict)

    def test_build_payload_method(self):
        """Test the _build_payload method."""
        llm = OllamaLlm(model="test-model")
        mock_request = MagicMock()
        mock_request.contents = []
        mock_request.generation_config = None
        mock_request.tools = None
        
        payload = llm._build_payload(mock_request, stream=False)
        assert isinstance(payload, dict)
        assert "model" in payload
        assert "stream" in payload
        assert payload["stream"] is False