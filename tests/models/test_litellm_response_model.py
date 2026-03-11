from unittest.mock import MagicMock, patch

from minisweagent.models.litellm_model import DEFAULT_LITELLM_TIMEOUT_SECONDS
from minisweagent.models.litellm_response_model import LitellmResponseModel


def _mock_litellm_response_output():
    response = MagicMock()
    response.output = [
        {"type": "function_call", "call_id": "call_resp_1", "name": "bash", "arguments": '{"command": "pwd"}'}
    ]
    response.model_dump.return_value = {
        "object": "response",
        "output": response.output,
    }
    return response


@patch("minisweagent.models.litellm_response_model.litellm.responses")
@patch("minisweagent.models.litellm_response_model.litellm.cost_calculator.completion_cost")
def test_response_model_uses_default_timeout_when_unspecified(mock_cost, mock_responses):
    mock_responses.return_value = _mock_litellm_response_output()
    mock_cost.return_value = 0.001

    model = LitellmResponseModel(model_name="gpt-5-mini")
    model.query([{"role": "user", "content": "test"}])

    assert mock_responses.call_args.kwargs["timeout"] == DEFAULT_LITELLM_TIMEOUT_SECONDS


@patch("minisweagent.models.litellm_response_model.litellm.responses")
@patch("minisweagent.models.litellm_response_model.litellm.cost_calculator.completion_cost")
def test_response_model_preserves_explicit_timeout(mock_cost, mock_responses):
    mock_responses.return_value = _mock_litellm_response_output()
    mock_cost.return_value = 0.001

    model = LitellmResponseModel(model_name="gpt-5-mini", model_kwargs={"timeout": 30})
    model.query([{"role": "user", "content": "test"}])

    assert mock_responses.call_args.kwargs["timeout"] == 30
