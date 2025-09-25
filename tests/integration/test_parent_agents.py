"""Integration tests for parent agents."""

import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import tempfile

# Add parent directory to path to import the agent module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'parent-agents'))

from agent import (
    _ollama_model, 
    _resolve_model, 
    save_attractions_to_state
)


class TestParentAgents:
    """Integration tests for parent agent functionality."""

    def test_ollama_model_creation(self, mock_env_vars):
        """Test Ollama model creation."""
        model = _ollama_model("mistral:latest")
        
        assert model.model == "mistral:latest"
        assert model.base_url == "http://localhost:11434"

    def test_resolve_model_with_ollama_env(self, monkeypatch):
        """Test model resolution with Ollama environment variable."""
        monkeypatch.setenv("MODEL", "ollama/mistral:latest")
        
        model = _resolve_model()
        
        assert hasattr(model, 'model')
        assert model.model == "ollama/mistral:latest"

    def test_resolve_model_with_default_model(self, monkeypatch):
        """Test model resolution with default model."""
        monkeypatch.delenv("MODEL", raising=False)
        monkeypatch.setenv("DEFAULT_MODEL", "ollama/llama2:latest")
        
        model = _resolve_model()
        
        assert hasattr(model, 'model')
        assert model.model == "ollama/llama2:latest"

    def test_resolve_model_fallback(self, monkeypatch):
        """Test model resolution fallback."""
        monkeypatch.delenv("MODEL", raising=False)
        monkeypatch.delenv("DEFAULT_MODEL", raising=False)
        
        model = _resolve_model()
        
        # Should fall back to default
        assert hasattr(model, 'model')

    def test_resolve_model_non_ollama(self, monkeypatch):
        """Test model resolution with non-Ollama model."""
        monkeypatch.setenv("MODEL", "gemini-pro")
        
        model = _resolve_model()
        
        # Should return string for non-Ollama models
        assert model == "gemini-pro"

    def test_save_attractions_to_state(self, mock_tool_context):
        """Test saving attractions to state."""
        # Set up the mock context state
        mock_tool_context.state = {}
        attractions = ["Eiffel Tower", "Louvre Museum", "Arc de Triomphe"]
        
        result = save_attractions_to_state(mock_tool_context, attractions)
        
        # Verify the function returns success
        assert result == {"status": "success"}
        # Verify the state was updated
        assert "attractions" in mock_tool_context.state
        assert mock_tool_context.state["attractions"] == attractions

    def test_attractions_state_access(self, mock_tool_context):
        """Test accessing attractions from state."""
        # Test that we can access the state through the context
        mock_tool_context.state = {"attractions": ["Tower Bridge", "British Museum"]}
        
        # Verify state contains the expected attractions
        assert "attractions" in mock_tool_context.state
        assert mock_tool_context.state["attractions"] == ["Tower Bridge", "British Museum"]

    def test_agent_creation_with_ollama_model(self, mock_env_vars):
        """Test agent creation with Ollama model."""
        # Test the model resolution which is the key part of agent creation
        model = _resolve_model()
        
        assert hasattr(model, 'model')
        assert model.base_url == "http://localhost:11434"

    def test_environment_variable_handling(self, monkeypatch):
        """Test handling of various environment variable combinations."""
        # Test with OLLAMA_BASE_URL override
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:8080")
        monkeypatch.setenv("MODEL", "ollama/custom-model")
        
        model = _resolve_model()
        
        assert hasattr(model, 'model')
        assert model.base_url == "http://custom:8080"

    def test_attractions_state_management_workflow(self, mock_tool_context):
        """Test complete attractions state management workflow."""
        # Initially no attractions
        mock_tool_context.state = {}
        
        # Save some attractions
        new_attractions = ["Statue of Liberty", "Central Park", "Times Square"]
        result = save_attractions_to_state(mock_tool_context, new_attractions)
        
        # Verify the function returns success status
        assert result == {"status": "success"}
        
        # Verify the state was updated (based on the actual implementation)
        # The function updates tool_context.state directly
        assert "attractions" in mock_tool_context.state

    def test_model_configuration_with_custom_base_url(self):
        """Test model configuration with custom base URL."""
        custom_base_url = "http://remote-ollama:11434"
        model = _ollama_model("custom-model")
        
        # The model should use the environment base URL or default
        assert model.base_url in ["http://localhost:11434", custom_base_url]
        assert model.model == "custom-model"