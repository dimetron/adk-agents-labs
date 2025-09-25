"""Integration tests for workflow agents."""

import os
import pytest
from unittest.mock import MagicMock, patch
import sys

# Add parent directory to path to import the agent module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'workflow-agents'))


class TestWorkflowAgents:
    """Integration tests for workflow agent functionality."""

    def test_ollama_model_creation(self, mock_env_vars):
        """Test Ollama model creation."""
        try:
            from agent import _ollama_model
            model = _ollama_model("mistral:latest")
            assert model.model == "mistral:latest"
            assert model.base_url == "http://localhost:11434"
        except ImportError:
            pytest.skip("Workflow agents module not available")

    def test_resolve_model_with_ollama_env(self, monkeypatch):
        """Test model resolution with Ollama environment variable."""
        monkeypatch.setenv("MODEL", "ollama/mistral:latest")
        
        try:
            from agent import _resolve_model
            model = _resolve_model()
            assert hasattr(model, 'model')
            assert model.model == "ollama/mistral:latest"
        except ImportError:
            pytest.skip("Workflow agents module not available")

    def test_file_operations(self):
        """Test file operations."""
        try:
            from agent import write_file, append_to_state
            
            # Test write_file function
            mock_context = MagicMock()
            directory = "movie_pitches"
            filename = "test_movie.txt"
            content = "This is a test movie pitch."
            
            with patch('os.makedirs'), \
                 patch('builtins.open', create=True) as mock_open:
                
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                result = write_file(mock_context, directory, filename, content)
                assert result == {"status": "success"}
                mock_file.write.assert_called_once_with(content)
                
            # Test append_to_state function
            mock_context.state = {}
            result = append_to_state(mock_context, "movie_ideas", "Great movie idea")
            assert result == {"status": "success"}
            assert "movie_ideas" in mock_context.state
                
        except ImportError:
            pytest.skip("Workflow agents module not available")

    def test_langchain_integration(self):
        """Test that langchain tools are available in the workflow."""
        try:
            # Test that we can import the langchain components
            from langchain_community.tools import WikipediaQueryRun
            from langchain_community.utilities import WikipediaAPIWrapper
            
            # This tests that the imports work, which is part of the workflow agent setup
            assert WikipediaQueryRun is not None
            assert WikipediaAPIWrapper is not None
            
        except ImportError:
            pytest.skip("Langchain components not available")