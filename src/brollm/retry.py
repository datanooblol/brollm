"""Retry helper meant to wrap an `input_fn` (or any callable), not `BaseContract`.

Retrying is a concern of *how a call is fetched*, which the contract already
hands to `input_fn` — so this stays a plain decorator you apply yourself
instead of a hidden behavior of `BaseContract.run()`.

    @retry(max_attempts=3, on=(TimeoutError, ConnectionError))
    def input_fn(system_prompt, messages):
        return provider_client.call(system_prompt, messages)

    llm = BaseContract(input_fn=input_fn, output_fn=output_fn)

Design decisions baked into this first pass (flag any of these to change):

- Default `on=(Exception,)` — retries anything, since brollm has no idea
  what exceptions any given provider raises. Narrow it per-provider via
  `on=(...)`.
- Backoff defaults to exponential (`base_delay * 2 ** (attempt - 1)`) with
  jitter on, since fixed-delay retries against a rate-limited API tend to
  synchronize failures. Pass `backoff="fixed"` or your own
  `Callable[[int], float]` to override.
- `on_retry(attempt, exception)` is an optional hook for logging/metrics —
  called before sleeping, not called on the final (raising) attempt.
- Synchronous only. No async variant yet.
"""

from collections.abc import Callable
from functools import wraps
from random import uniform
from time import sleep

Backoff = str | Callable[[int], float]


def _delay_for(attempt: int, base_delay: float, backoff: Backoff, jitter: bool) -> float:
    """Computes the delay before the next retry attempt.

    Args:
        attempt: The 1-indexed attempt number that just failed.
        base_delay: Base delay in seconds, interpreted per `backoff`.
        backoff: `"fixed"`, `"exponential"`, or a callable mapping `attempt`
            to a delay in seconds.
        jitter: If `True`, scales the computed delay by a random factor in
            `[0.5, 1.5)` to avoid synchronized retry storms.

    Returns:
        The delay in seconds to sleep before the next attempt.

    Raises:
        ValueError: If `backoff` is a string other than `"fixed"` or
            `"exponential"`.
    """
    if callable(backoff):
        delay = backoff(attempt)
    elif backoff == "fixed":
        delay = base_delay
    elif backoff == "exponential":
        delay = base_delay * (2 ** (attempt - 1))
    else:
        raise ValueError(f"Unknown backoff: {backoff!r}")

    if jitter:
        delay *= uniform(0.5, 1.5)
    return delay


def retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    backoff: Backoff = "exponential",
    on: tuple[type[BaseException], ...] = (Exception,),
    jitter: bool = True,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> Callable[[Callable], Callable]:
    """Builds a decorator that retries a callable on matching exceptions.

    Args:
        max_attempts: Maximum number of attempts (including the first),
            before the last exception is re-raised.
        base_delay: Base delay in seconds passed to `backoff`.
        backoff: `"fixed"`, `"exponential"`, or a callable mapping the
            failed attempt number (1-indexed) to a delay in seconds.
        on: Exception types that trigger a retry. Any other exception
            propagates immediately, untouched.
        jitter: If `True`, randomizes each computed delay to avoid
            synchronized retry storms.
        on_retry: Optional callback invoked as `on_retry(attempt, exc)`
            after a failed attempt but before sleeping. Not called on the
            final, raising attempt.

    Returns:
        A decorator that wraps a callable with the retry behavior above.
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                attempt += 1
                try:
                    return fn(*args, **kwargs)
                except on as exc:
                    if attempt >= max_attempts:
                        raise
                    if on_retry is not None:
                        on_retry(attempt, exc)
                    sleep(_delay_for(attempt, base_delay, backoff, jitter))
        return wrapper
    return decorator
