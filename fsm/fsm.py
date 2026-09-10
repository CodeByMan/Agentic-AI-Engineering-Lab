import json
import os
from enum import StrEnum
from pathlib import Path


class State(StrEnum):
    INIT = "INIT"
    READING = "READING"
    ANALYZING = "ANALYZING"
    DONE = "DONE"
    FAILED = "FAILED"


class PersistentFSM:
    TRANSITIONS = {
        State.INIT: {State.READING},
        State.READING: {State.ANALYZING, State.FAILED},
        State.ANALYZING: {State.DONE, State.FAILED},
        State.DONE: set(),
        State.FAILED: set(),
    }

    def __init__(self, db_path: str = "checkpoint.json"):
        self.db_path = Path(db_path)
        self.state = State.INIT
        self.retries = 0
        self.recover()  # Recover automatically after a previous crash.

    def recover(self) -> None:
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text(encoding="utf-8"))
                self.state = State(data["state"])
                self.retries = int(data["retries"])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"FSM checkpoint is unreadable: {self.db_path}") from exc

    def _checkpoint(self) -> None:
        """Persist the current state synchronously before continuing."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.db_path.with_name(f"{self.db_path.name}.tmp")
        temporary_path.write_text(
            json.dumps({"state": self.state.value, "retries": self.retries}),
            encoding="utf-8",
        )
        os.replace(temporary_path, self.db_path)

    def transition_to(self, new_state: State, guard: bool = True) -> None:
        # 2. Invariant: reject transitions outside the transition matrix.
        if new_state not in self.TRANSITIONS[self.state]:
            raise ValueError(f"Illegal transition: {self.state} -> {new_state}")

        # 3. Evaluate the transition guard.
        if not guard:
            raise PermissionError(f"Guard failed for target state: {new_state}")

        self.state = new_state
        self._checkpoint()  # Persist the new state before the next step.


# --- Performance and crash-recovery demo ---
if __name__ == "__main__":
    fsm = PersistentFSM()
    print(f"[BOOT] Current state: {fsm.state}")

    if fsm.state == State.INIT:
        fsm.transition_to(State.READING)
        print(f"[TRANSITION] Moved to: {fsm.state}")

        # Simulate a power loss in the middle of the workflow.
        print("[CRASH] Simulating power loss...")
        raise SystemExit(1)
    elif fsm.state == State.READING:
        fsm.transition_to(State.ANALYZING)
        print(f"[RECOVERY] Resumed at: {fsm.state}")
        fsm.transition_to(State.DONE)
        print(f"[DONE] Workflow completed at: {fsm.state}")
    elif fsm.state == State.DONE:
        print("[DONE] Workflow was already completed.")
    elif fsm.state == State.FAILED:
        print("[FAILED] Workflow requires explicit recovery.")
