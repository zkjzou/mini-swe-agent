import logging
from types import SimpleNamespace

from tenacity.wait import wait_random_exponential

from minisweagent.models.utils.retry import retry


def test_retry_uses_jittered_exponential_backoff_with_expected_bounds():
    retrying = retry(logger=logging.getLogger("test-retry"), abort_exceptions=[])

    assert isinstance(retrying.wait, wait_random_exponential)
    assert retrying.wait.multiplier == 1
    assert retrying.wait.min == 4.0
    assert retrying.wait.max == 120.0
    assert retrying.wait.exp_base == 2


def test_retry_respects_attempt_override(monkeypatch):
    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "7")

    retrying = retry(logger=logging.getLogger("test-retry"), abort_exceptions=[])

    assert retrying.stop.max_attempt_number == 7


def test_retry_wait_has_floor_and_cap():
    retrying = retry(logger=logging.getLogger("test-retry"), abort_exceptions=[])

    for attempt_number in range(1, 20):
        retry_state = SimpleNamespace(attempt_number=attempt_number)
        wait_time = retrying.wait(retry_state)
        assert wait_time >= 4.0
        assert wait_time <= 120.0
