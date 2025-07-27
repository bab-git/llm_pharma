"""
Unit tests for the LLMManager module.
"""

from unittest.mock import patch

import pytest

from backend.my_agent.llm_manager import LLMManager


class TestLLMManager:
    """Test cases for LLMManager class."""

    def test_llm_manager_initialization(self):
        """Test LLMManager initialization with model configs."""
        model_configs = [
            ("test-model-1", "groq"),
            ("test-model-2", "groq"),
        ]

        with patch("backend.my_agent.llm_manager.ChatGroq"):
            manager = LLMManager(model_configs, temperature=0.5)

            assert manager.model_configs == model_configs
            assert manager.temperature == 0.5
            assert manager.current_index == 0
            assert len(manager.clients) == 2

    def test_llm_manager_initialization_with_openai(self):
        """Test LLMManager initialization with OpenAI provider."""
        model_configs = [
            ("gpt-3.5-turbo", "openai"),
        ]

        with patch("langchain_openai.ChatOpenAI"):
            manager = LLMManager(model_configs)

            assert manager.model_configs == model_configs
            assert manager.temperature == 0.0  # default
            assert manager.current_index == 0
            assert len(manager.clients) == 1

    def test_llm_manager_invalid_provider(self):
        """Test LLMManager raises ValueError for invalid provider."""
        model_configs = [
            ("test-model", "invalid_provider"),
        ]

        with pytest.raises(ValueError, match="Unknown provider: invalid_provider"):
            LLMManager(model_configs)

    def test_llm_manager_advance(self):
        """Test the advance method moves to next model."""
        model_configs = [
            ("test-model-1", "groq"),
            ("test-model-2", "groq"),
        ]

        with patch("backend.my_agent.llm_manager.ChatGroq"):
            manager = LLMManager(model_configs)

            # Start at index 0
            assert manager.current_index == 0

            # Advance to next model
            result = manager.advance()
            assert result is True
            assert manager.current_index == 1

            # Advance again (should return False when no more models)
            result = manager.advance()
            assert result is False
            assert manager.current_index == 1  # Should stay at last model

    def test_llm_manager_reset(self):
        """Test the reset method returns to first model."""
        model_configs = [
            ("test-model-1", "groq"),
            ("test-model-2", "groq"),
        ]

        with patch("backend.my_agent.llm_manager.ChatGroq"):
            manager = LLMManager(model_configs)

            # Move to second model
            manager.advance()
            assert manager.current_index == 1

            # Reset to first model
            manager.reset()
            assert manager.current_index == 0

    def test_llm_manager_current_model_id(self):
        """Test current_model_id property returns correct model ID."""
        model_configs = [
            ("test-model-1", "groq"),
            ("test-model-2", "groq"),
        ]

        with patch("backend.my_agent.llm_manager.ChatGroq"):
            manager = LLMManager(model_configs)

            # Should return first model ID
            assert manager.current_model_id == "test-model-1"

            # Advance and check second model ID
            manager.advance()
            assert manager.current_model_id == "test-model-2"

    @patch("backend.my_agent.llm_manager.ChatGroq")
    def test_get_default_managers(self, mock_chat_groq):
        """Test get_default_managers class method."""
        completion_manager, tool_manager = LLMManager.get_default_managers()

        assert isinstance(completion_manager, LLMManager)
        assert isinstance(tool_manager, LLMManager)
        assert len(completion_manager.model_configs) > 0
        assert len(tool_manager.model_configs) > 0
