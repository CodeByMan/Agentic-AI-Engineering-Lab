import time
import random
import hashlib
from threading import Lock
from typing import Any, Callable, Dict, Tuple


_IDEMPOTENCY_RESULTS: Dict[str, Dict[str, Any]] = {}
_IDEMPOTENCY_LOCKS: Dict[str, Lock] = {}
_LOCKS_GUARD = Lock()


def _key_lock(idempotency_key: str) -> Lock:
    with _LOCKS_GUARD:
        return _IDEMPOTENCY_LOCKS.setdefault(idempotency_key, Lock())


def execute_idempotent_activity(
    task_id: str,
    step_name: str,
    action_fn: Callable[[str], Any],
    max_retries: int = 3,
    retryable_exceptions: Tuple[type[Exception], ...] = (TimeoutError, ConnectionError),
    idempotency_store: Dict[str, Any] | None = None,
    scope_id: str | None = None,
):
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a non-empty string")
    if not isinstance(step_name, str) or not step_name.strip():
        raise ValueError("step_name must be a non-empty string")
    if not callable(action_fn):
        raise TypeError("action_fn must be callable")
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    if idempotency_store is None and (not isinstance(scope_id, str) or not scope_id.strip()):
        raise ValueError("scope_id is required when using the shared idempotency store")

    # 1. Generate a deterministic idempotency key.
    namespace = scope_id.strip() if isinstance(scope_id, str) and scope_id.strip() else "custom-store"
    idempotency_key = hashlib.sha256(
        f"{namespace}:{task_id}:{step_name}".encode()
    ).hexdigest()
    result_store = (
        _IDEMPOTENCY_RESULTS.setdefault(namespace, {})
        if idempotency_store is None
        else idempotency_store
    )

    with _key_lock(idempotency_key):
        if idempotency_key in result_store:
            print(f"[IDEMPOTENT REPLAY] Returning cached result for key {idempotency_key[:12]}.")
            return result_store[idempotency_key]

        base_delay = 0.5
        for attempt in range(max_retries):
            try:
                # The caller can attach this key to the request or execution layer.
                print(f"[ATTEMPT {attempt + 1}] Executing with Key: {idempotency_key[:12]}")
                result = action_fn(idempotency_key)
                result_store[idempotency_key] = result
                return result

            except retryable_exceptions as e:
                if attempt == max_retries - 1:
                    print(f"[PERMANENT FAILURE] Max retries reached for key {idempotency_key[:12]}.")
                    raise

                # Calculate full-jitter backoff while holding the per-key lock so
                # concurrent callers cannot duplicate the side effect.
                sleep_time = random.uniform(0, min(8.0, base_delay * (2 ** attempt)))
                print(
                    f"[TRANSIENT ERROR] {type(e).__name__} -> "
                    f"Retrying in {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)
            except Exception:
                raise


if __name__ == "__main__":
    attempts = 0

    def flaky_action(idempotency_key: str) -> str:
        nonlocal_attempts[0] += 1
        if nonlocal_attempts[0] < 3:
            raise ConnectionError("temporary downstream failure")
        return f"Completed with key {idempotency_key}"

    nonlocal_attempts = [attempts]
    print(
        execute_idempotent_activity(
            "demo-task", "demo-step", flaky_action, scope_id="demo-scope"
        )
    )
    print(
        execute_idempotent_activity(
            "demo-task", "demo-step", flaky_action, scope_id="demo-scope"
        )
    )
