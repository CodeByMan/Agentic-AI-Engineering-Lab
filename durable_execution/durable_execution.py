import json
import os
from pathlib import Path
from typing import Any, Callable

LOG_FILE = Path(__file__).resolve().with_name("execution_log.json")
CRASH_MARKER = "_demo_crash_injected"


def _read_log() -> dict[str, Any]:
    if not LOG_FILE.exists():
        return {"runs": {}}

    try:
        data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Durable log is unreadable: {LOG_FILE}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Durable log must contain an object")
    if "runs" not in data:
        return {"runs": {"default": data}}
    if not isinstance(data["runs"], dict):
        raise RuntimeError("Durable log must contain a 'runs' object")
    return data


def _write_log(log: dict[str, Any]) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = LOG_FILE.with_name(f"{LOG_FILE.name}.tmp")
    temporary_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    os.replace(temporary_path, LOG_FILE)


def _run_log(log: dict[str, Any], run_id: str) -> dict[str, Any]:
    runs = log.setdefault("runs", {})
    run = runs.setdefault(run_id, {})
    if not isinstance(run, dict):
        raise RuntimeError(f"Durable run '{run_id}' must contain an object")
    return run


def run_activity(
    step_id: str,
    func: Callable[..., Any],
    *args,
    run_id: str = "default",
):
    """Run an activity once per run and replay its JSON-serializable result."""
    if not step_id or not run_id:
        raise ValueError("step_id and run_id must be non-empty")

    log = _read_log()
    run = _run_log(log, run_id)

    # 1. Replay check: skip work that already has a recorded result.
    if step_id in run:
        print(f"[REPLAY] Step '{step_id}' skipped! Restored from cache.")
        return run[step_id]

    # 2. Execute the non-deterministic activity for the first time.
    print(f"[EXECUTE] Running '{step_id}'...")
    result = func(*args)

    # 3. Commit the result to disk.
    try:
        json.dumps(result)
    except TypeError as exc:
        raise TypeError("Activity results must be JSON serializable") from exc
    run[step_id] = result
    _write_log(log)
    return result


# --- Deterministic workflow orchestrator ---
def my_agent_workflow(run_id: str = "demo-workflow"):
    # Step 1: extract text.
    text = run_activity("step_1_read", lambda: "fake_pdf_content", run_id=run_id)

    # Step 2: call the expensive LLM activity.
    summary = run_activity("step_2_llm", lambda: f"Summary of {text}", run_id=run_id)

    # Simulate a single crash before step 3, then allow recovery on the next run.
    log = _read_log()
    run = _run_log(log, run_id)
    if "step_3_db" not in run and not run.get(CRASH_MARKER, False):
        run[CRASH_MARKER] = True
        _write_log(log)
        print("[CRASH] Power failure before step 3!")
        return False

    # Step 3: save to the database.
    db_status = run_activity(
        "step_3_db", lambda: f"Saved '{summary}' to DB", run_id=run_id
    )
    print(f"[DONE] Workflow completed: {db_status}")
    return True


if __name__ == "__main__":
    if not my_agent_workflow():
        raise SystemExit(1)
