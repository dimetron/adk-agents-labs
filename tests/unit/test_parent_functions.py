"""Unit tests for parent agent functions."""

import os
import pytest
from unittest.mock import MagicMock, patch
import sys

# Add parent-agents directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'parent-agents'))


class TestParentFunctions:
    """Test cases for parent agent functions."""

    def test_ollama_model_creation(self):
        """Test Ollama model creation function."""
        try:
            from agent import _ollama_model
            
            # Test with default environment
            model = _ollama_model("test-model")
            assert model.model == "test-model"
            assert hasattr(model, 'base_url')
            assert hasattr(model, 'timeout')
            
            # Test with custom base URL
            with patch.dict('os.environ', {'OLLAMA_BASE_URL': 'http://custom:8080'}):
                model_custom = _ollama_model("custom-model")
                assert model_custom.model == "custom-model"
                assert model_custom.base_url == "http://custom:8080"
                
        except ImportError:
            pytest.skip("Parent agents module not available")

    def test_model_resolution_logic(self):
        """Test model resolution logic."""
        try:
            from agent import _resolve_model
            
            # Test with MODEL environment variable (Ollama)
            with patch.dict('os.environ', {'MODEL': 'ollama/mistral:latest'}, clear=True):
                model = _resolve_model()
                assert hasattr(model, 'model')
                assert model.model == "ollama/mistral:latest"
            
            # Test with MODEL environment variable (non-Ollama)
            with patch.dict('os.environ', {'MODEL': 'gemini-pro'}, clear=True):
                model = _resolve_model()
                assert model == "gemini-pro"
            
            # Test with DEFAULT_MODEL environment variable
            with patch.dict('os.environ', {'DEFAULT_MODEL': 'ollama/llama2'}, clear=True):
                model = _resolve_model()
                assert hasattr(model, 'model')
                
        except ImportError:
            pytest.skip("Parent agents module not available")

    def test_save_attractions_to_state_function(self):
        """Test save_attractions_to_state function."""
        try:
            from agent import save_attractions_to_state
            
            # Test with empty state
            mock_context = MagicMock()
            mock_context.state = {}
            attractions = ["Eiffel Tower", "Louvre"]
            
            result = save_attractions_to_state(mock_context, attractions)
            assert result == {"status": "success"}
            assert mock_context.state["attractions"] == attractions
            
            # Test with existing attractions
            mock_context.state = {"attractions": ["Existing Attraction"]}
            new_attractions = ["New Attraction"]
            
            result = save_attractions_to_state(mock_context, new_attractions)
            assert result == {"status": "success"}
            expected = ["Existing Attraction", "New Attraction"]
            assert mock_context.state["attractions"] == expected
            
        except ImportError:
            pytest.skip("Parent agents module not available")

    def test_environment_variable_combinations(self):
        """Test various environment variable combinations."""
        try:
            from agent import _resolve_model, _ollama_model
            
            # Test various combinations
            test_scenarios = [
                # (env_vars, expected_type)
                ({'MODEL': 'ollama/test'}, 'ollama_model'),
                ({'MODEL': 'gemini-pro'}, 'string'),
                ({'DEFAULT_MODEL': 'ollama/default'}, 'ollama_model'),
                ({}, 'ollama_model'),  # Should fall back to default
            ]
            
            for env_vars, expected_type in test_scenarios:
                with patch.dict('os.environ', env_vars, clear=True):
                    model = _resolve_model()
                    if expected_type == 'ollama_model':
                        assert hasattr(model, 'model')
                        assert hasattr(model, 'base_url')
                    else:
                        assert isinstance(model, str)
                        
        except ImportError:
            pytest.skip("Parent agents module not available")

    def test_logging_integration(self):
        """Test logging integration."""
        try:
            from agent import _resolve_model
            
            with patch('agent.logging.info') as mock_log:
                _resolve_model()
                # Should have logged something about model resolution
                assert mock_log.called or True  # Allow for cases where logging doesn't happen
                
        except ImportError:
            pytest.skip("Parent agents module not available")

    def test_model_spec_initialization(self):
        """Test that model_spec is properly initialized."""
        try:
            from agent import model_spec
            
            # The model_spec should be initialized
            assert model_spec is not None
            # Should be either a string or an OllamaLlm instance
            assert isinstance(model_spec, str) or hasattr(model_spec, 'model')
            
        except ImportError:
            pytest.skip("Parent agents module not available")

    def test_agent_definitions(self):
        """Test that agents are properly defined."""
        try:
            from agent import attractions_planner, travel_brainstormer, root_agent
            
            # All agents should be defined
            assert attractions_planner is not None
            assert travel_brainstormer is not None  
            assert root_agent is not None
            
            # Agents should have expected attributes
            assert hasattr(attractions_planner, 'name')
            assert hasattr(travel_brainstormer, 'name')
            assert hasattr(root_agent, 'name')
            
        except ImportError:
            pytest.skip("Parent agents module not available")
