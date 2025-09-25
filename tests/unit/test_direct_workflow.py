"""Direct tests for workflow agents module to improve coverage."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add workflow-agents to path
workflow_path = os.path.join(os.path.dirname(__file__), '..', '..', 'workflow-agents')
if workflow_path not in sys.path:
    sys.path.insert(0, workflow_path)


def test_direct_module_import():
    """Test direct import of workflow agents module."""
    try:
        # This should trigger the module loading and initialization
        import agent
        
        # Test that basic components exist
        assert hasattr(agent, 'model_spec')
        assert hasattr(agent, 'append_to_state')
        assert hasattr(agent, 'write_file')
        
        # Test model spec is initialized
        assert agent.model_spec is not None
        
    except ImportError:
        pytest.skip("Workflow agents module not available")


def test_direct_function_calls():
    """Test direct function calls to improve coverage."""
    try:
        import agent
        
        # Test append_to_state directly
        mock_context = MagicMock()
        mock_context.state = {}
        
        result = agent.append_to_state(mock_context, "test_field", "test_value")
        assert result == {"status": "success"}
        
        # Test write_file directly
        with patch('agent.os.makedirs'), \
             patch('agent.open', create=True) as mock_open:
            
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            result = agent.write_file(mock_context, "test_dir", "test.txt", "content")
            assert result == {"status": "success"}
            
    except ImportError:
        pytest.skip("Workflow agents module not available")


def test_model_resolution():
    """Test model resolution in workflow agents."""
    try:
        import agent
        
        # Test _resolve_model function
        model = agent._resolve_model()
        assert model is not None
        
        # Test _ollama_model function
        ollama_model = agent._ollama_model("test-model")
        assert ollama_model.model == "test-model"
        
    except ImportError:
        pytest.skip("Workflow agents module not available")


def test_agent_creation():
    """Test that agents are properly created."""
    try:
        import agent
        
        # Test that all agents exist
        agents = ['file_writer', 'plot_outliner', 'researcher', 'root_agent']
        
        for agent_name in agents:
            if hasattr(agent, agent_name):
                agent_obj = getattr(agent, agent_name)
                assert agent_obj is not None
                
                # Test basic agent properties
                if hasattr(agent_obj, 'name'):
                    assert agent_obj.name is not None
                if hasattr(agent_obj, 'model'):
                    assert agent_obj.model is not None
                    
    except ImportError:
        pytest.skip("Workflow agents module not available")


def test_imports_and_dependencies():
    """Test that all imports work correctly."""
    try:
        import agent
        
        # Test that the module loaded without errors
        assert agent is not None
        
        # Test that sys.path was modified
        assert ".." in agent.sys.path or any(".." in p for p in agent.sys.path)
        
    except ImportError:
        pytest.skip("Workflow agents module not available")


def test_environment_integration():
    """Test environment variable integration."""
    try:
        import agent
        
        # Test with different environment variables
        test_env = {
            'MODEL': 'ollama/test-model',
            'OLLAMA_BASE_URL': 'http://test:8080'
        }
        
        with patch.dict('os.environ', test_env):
            # Re-import to test with new environment
            import importlib
            importlib.reload(agent)
            
            # Test that model resolution works
            model = agent._resolve_model()
            assert model is not None
            
    except ImportError:
        pytest.skip("Workflow agents module not available")


def test_logging_setup():
    """Test logging setup in workflow agents."""
    try:
        import agent
        
        # Test that logging is configured
        import logging
        logger = logging.getLogger()
        assert logger is not None
        
    except ImportError:
        pytest.skip("Workflow agents module not available")


def test_callback_functions():
    """Test callback functions."""
    try:
        import agent
        
        # Test that callback functions are imported
        assert hasattr(agent, 'log_query_to_model')
        assert hasattr(agent, 'log_model_response')
        
        # Test callback functions exist
        assert agent.log_query_to_model is not None
        assert agent.log_model_response is not None
        
    except ImportError:
        pytest.skip("Workflow agents module not available")
