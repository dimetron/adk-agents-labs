"""Unit tests for callback logging functionality."""

import logging
import pytest
from unittest.mock import MagicMock, patch

from callback_logging import log_query_to_model, log_model_response


class TestCallbackLogging:
    """Test cases for callback logging functions."""

    def test_log_query_to_model_with_text(self, caplog):
        """Test logging query with text content."""
        # Mock objects
        mock_context = MagicMock()
        mock_context.agent_name = "test-agent"
        
        mock_request = MagicMock()
        mock_content = MagicMock()
        mock_content.role = "user"
        
        mock_part = MagicMock()
        mock_part.text = "Hello, how are you?"
        mock_content.parts = [mock_part]
        
        mock_request.contents = [mock_content]
        
        # Test logging
        with caplog.at_level(logging.INFO):
            log_query_to_model(mock_context, mock_request)
        
        assert "[query to test-agent]: Hello, how are you?" in caplog.text

    def test_log_query_to_model_no_text(self, caplog):
        """Test logging query without text content."""
        mock_context = MagicMock()
        mock_context.agent_name = "test-agent"
        
        mock_request = MagicMock()
        mock_content = MagicMock()
        mock_content.role = "user"
        
        mock_part = MagicMock()
        mock_part.text = None  # No text content
        mock_content.parts = [mock_part]
        
        mock_request.contents = [mock_content]
        
        with caplog.at_level(logging.INFO):
            log_query_to_model(mock_context, mock_request)
        
        # Should not log anything
        assert "[query to test-agent]" not in caplog.text

    def test_log_query_to_model_non_user_role(self, caplog):
        """Test logging query with non-user role."""
        mock_context = MagicMock()
        mock_context.agent_name = "test-agent"
        
        mock_request = MagicMock()
        mock_content = MagicMock()
        mock_content.role = "assistant"  # Not user role
        
        mock_part = MagicMock()
        mock_part.text = "Some response"
        mock_content.parts = [mock_part]
        
        mock_request.contents = [mock_content]
        
        with caplog.at_level(logging.INFO):
            log_query_to_model(mock_context, mock_request)
        
        # Should not log non-user queries
        assert "[query to test-agent]" not in caplog.text

    def test_log_query_to_model_empty_contents(self, caplog):
        """Test logging query with empty contents."""
        mock_context = MagicMock()
        mock_context.agent_name = "test-agent"
        
        mock_request = MagicMock()
        mock_request.contents = []  # Empty contents
        
        with caplog.at_level(logging.INFO):
            log_query_to_model(mock_context, mock_request)
        
        # Should not log anything
        assert "[query to test-agent]" not in caplog.text

    def test_log_model_response_with_text(self, caplog):
        """Test logging model response with text content."""
        mock_context = MagicMock()
        mock_context.agent_name = "test-agent"
        
        mock_response = MagicMock()
        mock_content = MagicMock()
        
        mock_part = MagicMock()
        mock_part.text = "Here is my response"
        mock_part.function_call = None
        mock_content.parts = [mock_part]
        
        mock_response.content = mock_content
        
        with caplog.at_level(logging.INFO):
            log_model_response(mock_context, mock_response)
        
        assert "[response from test-agent]: Here is my response" in caplog.text

    def test_log_model_response_with_function_call(self, caplog):
        """Test logging model response with function call."""
        mock_context = MagicMock()
        mock_context.agent_name = "test-agent"
        
        mock_response = MagicMock()
        mock_content = MagicMock()
        
        mock_part = MagicMock()
        mock_part.text = None
        mock_function_call = MagicMock()
        mock_function_call.name = "search_function"
        mock_part.function_call = mock_function_call
        mock_content.parts = [mock_part]
        
        mock_response.content = mock_content
        
        with caplog.at_level(logging.INFO):
            log_model_response(mock_context, mock_response)
        
        assert "[function call from test-agent]: search_function" in caplog.text

    def test_log_model_response_no_content(self, caplog):
        """Test logging model response without content."""
        mock_context = MagicMock()
        mock_context.agent_name = "test-agent"
        
        mock_response = MagicMock()
        mock_response.content = None  # No content
        
        with caplog.at_level(logging.INFO):
            log_model_response(mock_context, mock_response)
        
        # Should not log anything
        assert "[response from test-agent]" not in caplog.text

    def test_log_model_response_no_parts(self, caplog):
        """Test logging model response without parts."""
        mock_context = MagicMock()
        mock_context.agent_name = "test-agent"
        
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.parts = []  # No parts
        
        mock_response.content = mock_content
        
        with caplog.at_level(logging.INFO):
            log_model_response(mock_context, mock_response)
        
        # Should not log anything
        assert "[response from test-agent]" not in caplog.text

    def test_log_model_response_multiple_parts(self, caplog):
        """Test logging model response with multiple parts."""
        mock_context = MagicMock()
        mock_context.agent_name = "test-agent"
        
        mock_response = MagicMock()
        mock_content = MagicMock()
        
        # Create multiple parts
        mock_part1 = MagicMock()
        mock_part1.text = "First part"
        mock_part1.function_call = None
        
        mock_part2 = MagicMock()
        mock_part2.text = None
        mock_function_call = MagicMock()
        mock_function_call.name = "test_function"
        mock_part2.function_call = mock_function_call
        
        mock_part3 = MagicMock()
        mock_part3.text = "Third part"
        mock_part3.function_call = None
        
        mock_content.parts = [mock_part1, mock_part2, mock_part3]
        mock_response.content = mock_content
        
        with caplog.at_level(logging.INFO):
            log_model_response(mock_context, mock_response)
        
        # Should log both text and function call
        assert "[response from test-agent]: First part" in caplog.text
        assert "[function call from test-agent]: test_function" in caplog.text
        assert "[response from test-agent]: Third part" in caplog.text