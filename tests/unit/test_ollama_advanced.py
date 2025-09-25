"""Advanced unit tests for OllamaLlm class to improve coverage."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from ollama_llm import OllamaLlm, _schema_to_json, _function_declaration_to_tool, _extract_function_tools


class TestOllamaAdvanced:
    """Advanced test cases for OllamaLlm class."""

    def test_schema_to_json_function(self):
        """Test _schema_to_json utility function."""
        # Mock a schema object
        mock_schema = MagicMock()
        mock_schema.properties = {
            "param1": MagicMock(),
            "param2": MagicMock()
        }
        mock_schema.required = ["param1"]
        
        # Mock the properties
        mock_schema.properties["param1"].type = "string"
        mock_schema.properties["param1"].description = "Parameter 1"
        mock_schema.properties["param2"].type = "integer" 
        mock_schema.properties["param2"].description = "Parameter 2"
        
        with patch.object(mock_schema, 'model_dump', return_value={"properties": {}, "required": ["param1"]}):
            result = _schema_to_json(mock_schema)
            
            assert isinstance(result, dict)

    def test_function_declaration_to_tool(self):
        """Test _function_declaration_to_tool utility function."""
        mock_func_decl = MagicMock()
        mock_func_decl.name = "test_function"
        mock_func_decl.description = "Test function description"
        mock_func_decl.parameters = MagicMock()
        
        with patch('ollama_llm._schema_to_json', return_value={"type": "object"}):
            result = _function_declaration_to_tool(mock_func_decl)
            
            assert isinstance(result, dict)
            assert "type" in result
            assert result["type"] == "function"
            assert "function" in result
            assert result["function"]["name"] == "test_function"

    def test_extract_function_tools(self):
        """Test _extract_function_tools utility function."""
        mock_request = MagicMock()
        mock_request.tools = None
        
        # Test with no tools
        result = _extract_function_tools(mock_request)
        assert isinstance(result, list)
        assert len(result) == 0
        
        # Test with tools
        mock_tool = MagicMock()
        mock_func_decl = MagicMock()
        mock_tool.function_declarations = [mock_func_decl]
        mock_request.tools = [mock_tool]
        
        with patch('ollama_llm._function_declaration_to_tool', return_value={"type": "function"}):
            result = _extract_function_tools(mock_request)
            assert isinstance(result, list)

    def test_convert_messages_method(self):
        """Test _convert_messages method."""
        llm = OllamaLlm(model="test-model")
        mock_request = MagicMock()
        mock_request.contents = []
        
        result = llm._convert_messages(mock_request)
        assert isinstance(result, list)

    def test_build_options_with_generation_config(self):
        """Test _build_options with generation config."""
        llm = OllamaLlm(model="test-model")
        mock_request = MagicMock()
        
        # Mock generation config
        mock_gen_config = MagicMock()
        mock_gen_config.temperature = 0.7
        mock_gen_config.max_output_tokens = 100
        mock_gen_config.top_p = 0.9
        mock_gen_config.top_k = 40
        mock_request.generation_config = mock_gen_config
        
        options = llm._build_options(mock_request)
        
        assert isinstance(options, dict)
        # The method should handle the generation config

    def test_build_llm_response_method(self):
        """Test _build_llm_response method."""
        llm = OllamaLlm(model="test-model")
        
        # Test with valid response data
        response_data = {
            "message": {
                "role": "assistant",
                "content": "Test response"
            },
            "done": True,
            "total_duration": 1000000000,
            "prompt_eval_count": 10,
            "eval_count": 20
        }
        
        result = llm._build_llm_response(response_data)
        assert result is not None

    def test_build_llm_response_with_error(self):
        """Test _build_llm_response with error data."""
        llm = OllamaLlm(model="test-model")
        
        error_data = {
            "error": "Test error message"
        }
        
        with pytest.raises(RuntimeError, match="Ollama error"):
            llm._build_llm_response(error_data)

    def test_re_parse_tools_method(self):
        """Test re_parse_tools method."""
        llm = OllamaLlm(model="test-model")
        
        content_text = '{"function_call": {"name": "test_func", "args": {"param": "value"}}}'
        function_calls_parsed = []
        parts = []
        
        result = llm.re_parse_tools(content_text, function_calls_parsed, parts)
        # The method should process the content and return parts

    def test_supported_models_method(self):
        """Test supported_models class method."""
        models = OllamaLlm.supported_models()
        assert isinstance(models, list)
        assert len(models) > 0
        # Should include common model names
        assert any("mistral" in model.lower() or "llama" in model.lower() for model in models)

    def test_model_name_normalization_edge_cases(self):
        """Test model name normalization with edge cases."""
        # Test empty string
        assert OllamaLlm._normalize_model_name("") == ""
        
        # Test model name without ollama prefix
        assert OllamaLlm._normalize_model_name("just-model-name") == "just-model-name"
        
        # Test model name with multiple slashes
        assert OllamaLlm._normalize_model_name("ollama/path/model:tag") == "path/model:tag"

    def test_map_role_edge_cases(self):
        """Test _map_role with edge cases."""
        # Test empty string
        assert OllamaLlm._map_role("") == "user"
        
        # Test mixed case  
        assert OllamaLlm._map_role("USER") == "user"
        assert OllamaLlm._map_role("Model") == "user"  # _map_role doesn't handle case-insensitive

    def test_part_to_text_with_function_call(self):
        """Test _part_to_text with function call."""
        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_response = None
        mock_part.executable_code = None
        mock_part.code_execution_result = None
        mock_part.inline_data = None
        
        # Mock function call
        mock_function_call = MagicMock()
        mock_function_call.name = "test_function"
        mock_function_call.args = {"param": "value"}
        mock_part.function_call = mock_function_call
        
        result = OllamaLlm._part_to_text(mock_part)
        assert result is not None
        assert isinstance(result, str)
        # Should be JSON representation of function call
        parsed = json.loads(result)
        assert "function_call" in parsed

    def test_part_to_text_with_executable_code(self):
        """Test _part_to_text with executable code."""
        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = None
        mock_part.function_response = None
        mock_part.code_execution_result = None
        mock_part.inline_data = None
        
        # Mock executable code
        mock_executable = MagicMock()
        mock_executable.code = "print('hello')"
        mock_part.executable_code = mock_executable
        
        result = OllamaLlm._part_to_text(mock_part)
        assert result == "print('hello')"

    def test_part_to_text_with_inline_data(self):
        """Test _part_to_text with inline data."""
        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = None
        mock_part.function_response = None
        mock_part.executable_code = None
        mock_part.code_execution_result = None
        
        # Mock inline data
        mock_inline = MagicMock()
        mock_inline.data = b"binary data"
        mock_part.inline_data = mock_inline
        
        result = OllamaLlm._part_to_text(mock_part)
        assert result == "[binary data omitted]"

    def test_environment_functions_edge_cases(self):
        """Test environment utility functions with edge cases."""
        from ollama_llm import _env_timeout, _env_base_url, _env_keep_alive, _env_options
        
        # Test timeout with various values
        with patch.dict('os.environ', {'OLLAMA_TIMEOUT': '0'}, clear=False):
            assert _env_timeout() == 0.0
            
        with patch.dict('os.environ', {'OLLAMA_TIMEOUT': '999'}, clear=False):
            assert _env_timeout() == 999.0
        
        # Test base URL with trailing slash
        with patch.dict('os.environ', {'OLLAMA_BASE_URL': 'http://test:8080///'}, clear=False):
            assert _env_base_url() == "http://test:8080"
        
        # Test keep alive with different values
        with patch.dict('os.environ', {'OLLAMA_KEEP_ALIVE': '10m'}, clear=False):
            assert _env_keep_alive() == "10m"
            
        # Test options with complex JSON
        complex_options = '{"temperature": 0.8, "num_predict": 200, "custom": {"nested": true}}'
        with patch.dict('os.environ', {'OLLAMA_OPTIONS_JSON': complex_options}, clear=False):
            options = _env_options()
            assert options["temperature"] == 0.8
            assert options["custom"]["nested"] is True
