"""Unit tests for workflow agent functions."""

import os
import pytest
from unittest.mock import MagicMock, patch
import sys

# Add workflow-agents directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'workflow-agents'))


class TestWorkflowFunctions:
    """Test cases for workflow agent functions."""

    def test_append_to_state_function(self):
        """Test append_to_state function."""
        try:
            from agent import append_to_state
            
            mock_context = MagicMock()
            mock_context.state = {}
            
            # Test appending to new field
            result = append_to_state(mock_context, "ideas", "First idea")
            assert result == {"status": "success"}
            assert "ideas" in mock_context.state
            assert mock_context.state["ideas"] == ["First idea"]
            
            # Test appending to existing field
            mock_context.state = {"ideas": ["First idea"]}
            result = append_to_state(mock_context, "ideas", "Second idea")
            assert result == {"status": "success"}
            assert mock_context.state["ideas"] == ["First idea", "Second idea"]
            
        except ImportError:
            pytest.skip("Workflow agents module not available")

    def test_write_file_function(self):
        """Test write_file function."""
        try:
            from agent import write_file
            
            mock_context = MagicMock()
            directory = "test_dir"
            filename = "test_file.txt"
            content = "Test content"
            
            with patch('os.makedirs') as mock_makedirs, \
                 patch('builtins.open', create=True) as mock_open:
                
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                result = write_file(mock_context, directory, filename, content)
                
                assert result == {"status": "success"}
                mock_makedirs.assert_called_once()
                mock_file.write.assert_called_once_with(content)
                
        except ImportError:
            pytest.skip("Workflow agents module not available")

    def test_model_resolution_functions(self):
        """Test model resolution functions."""
        try:
            from agent import _ollama_model, _resolve_model
            
            # Test _ollama_model function
            model = _ollama_model("test-model")
            assert model.model == "test-model"
            assert hasattr(model, 'base_url')
            
            # Test _resolve_model with environment variables
            with patch.dict('os.environ', {'MODEL': 'ollama/test-model'}):
                resolved_model = _resolve_model()
                assert hasattr(resolved_model, 'model')
                
        except ImportError:
            pytest.skip("Workflow agents module not available")

    def test_environment_variable_handling(self):
        """Test environment variable handling in workflow agents."""
        try:
            from agent import _resolve_model
            
            # Test with different environment configurations
            test_cases = [
                {'MODEL': 'ollama/mistral:latest'},
                {'DEFAULT_MODEL': 'ollama/llama2'},
                {'MODEL': 'gemini-pro'},  # Non-Ollama model
            ]
            
            for env_vars in test_cases:
                with patch.dict('os.environ', env_vars, clear=True):
                    model = _resolve_model()
                    assert model is not None
                    
        except ImportError:
            pytest.skip("Workflow agents module not available")

    def test_workflow_agent_initialization(self):
        """Test that workflow agent components can be initialized."""
        try:
            # Test that we can import the main components
            from agent import model_spec
            assert model_spec is not None
            
        except ImportError:
            pytest.skip("Workflow agents module not available")
            
    def test_langchain_tool_integration(self):
        """Test langchain tool integration."""
        try:
            from langchain_community.tools import WikipediaQueryRun
            from langchain_community.utilities import WikipediaAPIWrapper
            
            # Test that we can create the Wikipedia tools
            wrapper = WikipediaAPIWrapper()
            tool = WikipediaQueryRun(api_wrapper=wrapper)
            
            assert wrapper is not None
            assert tool is not None
            
        except ImportError:
            pytest.skip("Langchain components not available")
