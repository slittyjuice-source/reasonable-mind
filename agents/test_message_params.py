#!/usr/bin/env python3
"""Test suite for Agent message_params functionality.

This module tests the ability to pass custom parameters to the Claude API
through the Agent's message_params argument, including headers, metadata,
and API parameters.
"""

import os
import sys

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent import Agent, ModelConfig


pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY is required for live API calls",
)


class TestMessageParams:
    __test__ = False  # Prevent pytest collection (requires live API key)
    """Test cases for message_params functionality."""

    def setup_method(self) -> None:
        """Initialize per-test attributes without blocking test collection."""
        self.verbose = True

    def _print(self, message: str) -> None:
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)

    def test_basic_agent(self) -> None:
        """Test agent without message_params to ensure backward compatibility."""
        agent = Agent(
            name="BasicAgent",
            system="You are a helpful assistant. Be very brief.",
            verbose=False
        )

        response = agent.run("What is 2+2?")
        # response is a list of message content blocks
        assert any("4" in str(block.get("text", "")) for block in response if block.get("type") == "text")
        response_text = next((block["text"] for block in response if block.get("type") == "text"), "")
        self._print(f"Response: {response_text}")

    def test_custom_headers(self) -> None:
        """Test passing custom headers through message_params."""
        agent = Agent(
            name="HeaderAgent",
            system="You are a helpful assistant. Be very brief.",
            verbose=False,
            message_params={
                "extra_headers": {
                    "X-Custom-Header": "test-value",
                    "X-Request-ID": "test-12345"
                }
            }
        )

        # Verify headers are stored
        assert "extra_headers" in agent.message_params
        assert agent.message_params["extra_headers"]["X-Custom-Header"] == "test-value"

        response = agent.run("What is 3+3?")
        response_text = next((block["text"] for block in response if block.get("type") == "text"), "")
        assert "6" in response_text
        self._print(f"Response with custom headers: {response_text}")

    def test_beta_headers(self) -> None:
        """Test passing beta feature headers."""
        agent = Agent(
            name="BetaAgent",
            system="You are a helpful assistant. Be very brief.",
            verbose=False,
            message_params={
                "extra_headers": {
                    "anthropic-beta": "files-api-2025-04-14"
                }
            }
        )

        # The API call should succeed even with beta headers
        response = agent.run("What is 5*5?")
        response_text = next((block["text"] for block in response if block.get("type") == "text"), "")
        assert "25" in response_text
        self._print(f"Response with beta headers: {response_text}")

    def test_metadata(self) -> None:
        """Test passing valid metadata fields."""
        agent = Agent(
            name="MetadataAgent",
            system="You are a helpful assistant. Be very brief.",
            verbose=False,
            message_params={
                "metadata": {
                    "user_id": "test-user-123"
                }
            }
        )

        response = agent.run("What is 10/2?")
        response_text = next((block["text"] for block in response if block.get("type") == "text"), "")
        assert "5" in response_text
        self._print(f"Response with metadata: {response_text}")

    def test_api_parameters(self) -> None:
        """Test passing various API parameters."""
        agent = Agent(
            name="ParamsAgent",
            system="You are a helpful assistant.",
            verbose=False,
            message_params={
                "top_k": 10,
                "top_p": 0.95,
                "temperature": 0.7
            }
        )

        # Verify parameters are passed through
        params = agent._prepare_message_params()
        assert params["top_k"] == 10
        assert params["top_p"] == 0.95
        assert params["temperature"] == 0.7

        response = agent.run("Say 'test'")
        response_text = next((block["text"] for block in response if block.get("type") == "text"), "")
        assert response_text
        self._print(f"Response with custom params: {response_text}")

    def test_parameter_override(self) -> None:
        """Test that message_params override config defaults."""
        config = ModelConfig(
            temperature=1.0,
            max_tokens=100
        )

        agent = Agent(
            name="OverrideAgent",
            system="You are a helpful assistant.",
            config=config,
            verbose=False,
            message_params={
                "temperature": 0.5,  # Should override config
                "max_tokens": 200    # Should override config
            }
        )

        params = agent._prepare_message_params()
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 200
        self._print("Parameter override successful")

    def test_invalid_metadata_field(self) -> None:
        """Test that invalid metadata fields are properly rejected by the API."""
        agent = Agent(
            name="InvalidAgent",
            system="You are a helpful assistant.",
            verbose=False,
            message_params={
                "metadata": {
                    "user_id": "valid",
                    "invalid_field": "should-fail"
                }
            }
        )

        try:
            agent.run("Test")
            # Should not reach here
            raise AssertionError("Expected API error for invalid metadata field")
        except Exception as e:
            assert "invalid_request_error" in str(e) or "metadata" in str(e).lower()
            self._print(f"Correctly rejected invalid metadata: {type(e).__name__}")

    def test_combined_parameters(self) -> None:
        """Test combining multiple parameter types."""
        agent = Agent(
            name="CombinedAgent",
            system="You are a helpful assistant. Be very brief.",
            verbose=False,
            message_params={
                "extra_headers": {
                    "X-Test": "combined",
                    "anthropic-beta": "files-api-2025-04-14"
                },
                "metadata": {
                    "user_id": "combined-test"
                },
                "temperature": 0.8,
                "top_k": 5
            }
        )

        params = agent._prepare_message_params()
        assert params["extra_headers"]["X-Test"] == "combined"
        assert params["metadata"]["user_id"] == "combined-test"
        assert params["temperature"] == 0.8
        assert params["top_k"] == 5

        response = agent.run("What is 1+1?")
        response_text = next((block["text"] for block in response if block.get("type") == "text"), "")
        assert "2" in response_text
        self._print(f"Response with combined params: {response_text}")
