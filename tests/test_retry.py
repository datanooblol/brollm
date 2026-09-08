import pytest

from brollm.retry import retry


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    monkeypatch.setattr("brollm.retry.sleep", lambda seconds: None)


def test_succeeds_without_retry_when_first_attempt_works():
    calls = {"n": 0}

    @retry(max_attempts=3)
    def fn():
        calls["n"] += 1
        return "ok"

    assert fn() == "ok"
    assert calls["n"] == 1


def test_retries_until_success_within_max_attempts():
    calls = {"n": 0}

    @retry(max_attempts=3, on=(ValueError,))
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_raises_original_exception_after_exhausting_max_attempts():
    calls = {"n": 0}

    @retry(max_attempts=2, on=(ValueError,))
    def always_fails():
        calls["n"] += 1
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        always_fails()

    assert calls["n"] == 2


def test_non_matching_exception_propagates_immediately_without_retry():
    calls = {"n": 0}

    @retry(max_attempts=5, on=(ValueError,))
    def wrong_type():
        calls["n"] += 1
        raise TypeError("not retried")

    with pytest.raises(TypeError, match="not retried"):
        wrong_type()

    assert calls["n"] == 1


def test_on_retry_hook_called_for_each_failed_attempt_but_not_the_final_one():
    seen = []

    @retry(max_attempts=3, on=(ValueError,), on_retry=lambda attempt, exc: seen.append((attempt, str(exc))))
    def always_fails():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        always_fails()

    assert seen == [(1, "boom"), (2, "boom")]


def test_fixed_backoff_uses_base_delay_every_time(monkeypatch):
    delays = []
    monkeypatch.setattr("brollm.retry.sleep", lambda seconds: delays.append(seconds))

    @retry(max_attempts=3, base_delay=1.0, backoff="fixed", jitter=False, on=(ValueError,))
    def always_fails():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        always_fails()

    assert delays == [1.0, 1.0]


def test_exponential_backoff_doubles_each_attempt(monkeypatch):
    delays = []
    monkeypatch.setattr("brollm.retry.sleep", lambda seconds: delays.append(seconds))

    @retry(max_attempts=4, base_delay=1.0, backoff="exponential", jitter=False, on=(ValueError,))
    def always_fails():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        always_fails()

    assert delays == [1.0, 2.0, 4.0]


def test_callable_backoff_receives_attempt_number(monkeypatch):
    delays = []
    monkeypatch.setattr("brollm.retry.sleep", lambda seconds: delays.append(seconds))

    @retry(max_attempts=3, backoff=lambda attempt: attempt * 10, jitter=False, on=(ValueError,))
    def always_fails():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        always_fails()

    assert delays == [10, 20]


def test_unknown_backoff_string_raises_value_error(monkeypatch):
    @retry(max_attempts=2, backoff="linear", jitter=False, on=(ValueError,))
    def always_fails():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="Unknown backoff"):
        always_fails()


def test_jitter_scales_delay_within_expected_bounds(monkeypatch):
    delays = []
    monkeypatch.setattr("brollm.retry.sleep", lambda seconds: delays.append(seconds))
    monkeypatch.setattr("brollm.retry.uniform", lambda low, high: 1.5)

    @retry(max_attempts=2, base_delay=2.0, backoff="fixed", jitter=True, on=(ValueError,))
    def always_fails():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        always_fails()

    assert delays == [3.0]
