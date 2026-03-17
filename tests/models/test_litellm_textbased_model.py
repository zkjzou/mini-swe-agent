import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import litellm
import pytest
from tenacity import Retrying, retry_if_not_exception_type, stop_after_attempt, wait_none

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel


def test_authentication_error_enhanced_message():
    """Test that AuthenticationError gets enhanced with config set instruction."""
    model = LitellmTextbasedModel(model_name="gpt-4")

    # Create a mock exception that behaves like AuthenticationError
    original_error = Mock(spec=litellm.exceptions.AuthenticationError)
    original_error.message = "Invalid API key"

    with patch("litellm.completion") as mock_completion:
        # Make completion raise the mock error
        def side_effect(*args, **kwargs):
            raise litellm.exceptions.AuthenticationError("Invalid API key", llm_provider="openai", model="gpt-4")

        mock_completion.side_effect = side_effect

        with pytest.raises(litellm.exceptions.AuthenticationError) as exc_info:
            model._query([{"role": "user", "content": "test"}])

        # Check that the error message was enhanced
        assert "You can permanently set your API key with `mini-extra config set KEY VALUE`." in str(exc_info.value)


def test_model_registry_loading():
    """Test that custom model registry is loaded and registered when provided."""
    model_costs = {
        "my-custom-model": {
            "max_tokens": 4096,
            "input_cost_per_token": 0.0001,
            "output_cost_per_token": 0.0002,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(model_costs, f)
        registry_path = f.name

    try:
        with patch("litellm.utils.register_model") as mock_register:
            _model = LitellmTextbasedModel(model_name="my-custom-model", litellm_model_registry=Path(registry_path))

            # Verify register_model was called with the correct data
            mock_register.assert_called_once_with(model_costs)
    except Exception as e:
        print(e)
        raise e
    finally:
        Path(registry_path).unlink()


def test_model_registry_none():
    """Test that no registry loading occurs when litellm_model_registry is None."""
    with patch("litellm.register_model") as mock_register:
        _model = LitellmTextbasedModel(model_name="gpt-4", litellm_model_registry=None)

        # Verify register_model was not called
        mock_register.assert_not_called()


def test_model_registry_not_provided():
    """Test that no registry loading occurs when litellm_model_registry is not provided."""
    with patch("litellm.register_model") as mock_register:
        _model = LitellmTextbasedModel(model_name="gpt-4o")

        # Verify register_model was not called
        mock_register.assert_not_called()


def test_litellm_model_cost_tracking_ignore_errors():
    """Test that models work with cost_tracking='ignore_errors'."""
    model = LitellmTextbasedModel(model_name="gpt-4o", cost_tracking="ignore_errors")

    initial_cost = GLOBAL_MODEL_STATS.cost

    with patch("litellm.completion") as mock_completion:
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "```mswea_bash_command\necho test\n```"
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "```mswea_bash_command\necho test\n```",
        }
        mock_response.choices = [Mock(message=mock_message)]
        mock_response.model_dump.return_value = {"test": "response"}
        mock_completion.return_value = mock_response

        with patch("litellm.cost_calculator.completion_cost", side_effect=ValueError("Model not found")):
            messages = [{"role": "user", "content": "test"}]
            result = model.query(messages)

            assert result["content"] == "```mswea_bash_command\necho test\n```"
            assert result["extra"]["actions"] == [{"command": "echo test"}]
            assert GLOBAL_MODEL_STATS.cost == initial_cost


def test_litellm_model_cost_validation_zero_cost():
    """Test that zero cost raises error when cost tracking is enabled."""
    model = LitellmTextbasedModel(model_name="gpt-4o")

    with patch("litellm.completion") as mock_completion:
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.model_dump.return_value = {"test": "response"}
        mock_completion.return_value = mock_response

        with patch("litellm.cost_calculator.completion_cost", return_value=0.0):
            messages = [{"role": "user", "content": "test"}]

            with pytest.raises(RuntimeError) as exc_info:
                model.query(messages)

            assert "Cost must be > 0.0, got 0.0" in str(exc_info.value)
            assert "MSWEA_COST_TRACKING='ignore_errors'" in str(exc_info.value)


def _make_retrying_stub(*, attempts: int, abort_exceptions: list[type[Exception]]) -> Retrying:
    return Retrying(
        reraise=True,
        stop=stop_after_attempt(attempts),
        wait=wait_none(),
        retry=retry_if_not_exception_type(tuple(abort_exceptions)),
    )


def _mock_text_completion_response(content: str = "```mswea_bash_command\necho test\n```") -> Mock:
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = content
    mock_message.model_dump.return_value = {"role": "assistant", "content": content}
    mock_response.choices = [Mock(message=mock_message)]
    mock_response.model_dump.return_value = {"choices": [{"message": {"content": content}}]}
    return mock_response


def test_query_retries_closed_client_error_and_clears_litellm_cache():
    model = LitellmTextbasedModel(model_name="gpt-4o")
    cache = SimpleNamespace(cache_dict={"stale": object()}, ttl_dict={"stale": 1.0}, expiration_heap=[(1.0, "stale")])
    closed_session = SimpleNamespace(is_closed=True)
    initial_cost = GLOBAL_MODEL_STATS.cost
    first_error = RuntimeError("Connection error.")
    first_error.__cause__ = RuntimeError("Cannot send a request, as the client has been closed.")

    with (
        patch(
            "minisweagent.models.litellm_model.retry",
            new=lambda *, logger, abort_exceptions: _make_retrying_stub(attempts=2, abort_exceptions=abort_exceptions),
        ),
        patch.object(litellm, "in_memory_llm_clients_cache", cache),
        patch.object(litellm, "client_session", closed_session),
        patch("litellm.completion", side_effect=[first_error, _mock_text_completion_response()]) as mock_completion,
        patch("litellm.cost_calculator.completion_cost", return_value=0.001),
    ):
        result = model.query([{"role": "user", "content": "test"}])

        assert result["content"] == "```mswea_bash_command\necho test\n```"
        assert result["extra"]["actions"] == [{"command": "echo test"}]
        assert result["extra"]["cost"] == 0.001
        assert mock_completion.call_count == 2
        assert cache.cache_dict == {}
        assert cache.ttl_dict == {}
        assert cache.expiration_heap == []
        assert litellm.client_session is None
        assert GLOBAL_MODEL_STATS.cost == pytest.approx(initial_cost + 0.001)


def test_query_raw_leaves_litellm_cache_unchanged_for_non_stale_errors():
    model = LitellmTextbasedModel(model_name="gpt-4o")
    cache = SimpleNamespace(cache_dict={"fresh": object()}, ttl_dict={"fresh": 2.0}, expiration_heap=[(2.0, "fresh")])

    with (
        patch(
            "minisweagent.models.litellm_model.retry",
            new=lambda *, logger, abort_exceptions: _make_retrying_stub(attempts=1, abort_exceptions=abort_exceptions),
        ),
        patch.object(litellm, "in_memory_llm_clients_cache", cache),
        patch("litellm.completion", side_effect=RuntimeError("Different transport failure")),
    ):
        with pytest.raises(RuntimeError, match="Different transport failure"):
            model.query_raw([{"role": "user", "content": "test"}])

    assert cache.cache_dict.keys() == {"fresh"}
    assert cache.ttl_dict.keys() == {"fresh"}
    assert cache.expiration_heap == [(2.0, "fresh")]
