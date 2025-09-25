"""Pytest configuration and shared fixtures for the test suite."""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {
        "model": "mistral:latest",
        "created_at": "2025-09-25T12:00:00.000000Z",
        "message": {
            "role": "assistant",
            "content": "This is a test response from Ollama."
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": 1000000000,
        "load_duration": 100000000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 200000000,
        "eval_count": 20,
        "eval_duration": 300000000
    }

@pytest.fixture
def mock_streaming_response():
    """Mock streaming Ollama API response."""
    return [
        {
            "model": "mistral:latest",
            "created_at": "2025-09-25T12:00:00.000000Z",
            "message": {
                "role": "assistant",
                "content": "This is"
            },
            "done": False
        },
        {
            "model": "mistral:latest",
            "created_at": "2025-09-25T12:00:00.000000Z",
            "message": {
                "role": "assistant",
                "content": " a test"
            },
            "done": False
        },
        {
            "model": "mistral:latest",
            "created_at": "2025-09-25T12:00:00.000000Z",
            "message": {
                "role": "assistant",
                "content": " response."
            },
            "done": True,
            "done_reason": "stop",
            "total_duration": 1000000000,
            "prompt_eval_count": 10,
            "eval_count": 20
        }
    ]

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    test_env = {
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_TIMEOUT": "120",
        "DEFAULT_MODEL": "ollama/mistral:latest",
        "MODEL": "ollama/mistral:latest"
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    return test_env

@pytest.fixture
def sample_llm_request():
    """Create a sample LLM request for testing."""
    try:
        from google.genai import types
        from google.adk.models.llm_request import LlmRequest
        
        content = types.Content(
            role="user",
            parts=[types.Part(text="Hello, this is a test message.")]
        )
        
        return LlmRequest(
            contents=[content],
            model="ollama/mistral:latest"
        )
    except ImportError:
        # Fallback for when google.genai is not available
        return MagicMock()

@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient for testing."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "mistral:latest",
        "message": {"role": "assistant", "content": "Test response"},
        "done": True
    }
    mock_client.post.return_value = mock_response
    return mock_client

@pytest.fixture
def temp_movie_pitches_dir(tmp_path):
    """Create a temporary movie_pitches directory for testing."""
    movie_dir = tmp_path / "movie_pitches"
    movie_dir.mkdir()
    return movie_dir

@pytest.fixture
def mock_tool_context():
    """Mock tool context for testing."""
    mock_context = MagicMock()
    mock_context.state = {}
    mock_context.save_to_state = MagicMock()
    mock_context.get_from_state = MagicMock(return_value=None)
    return mock_context
