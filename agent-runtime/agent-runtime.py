from dataclasses import dataclass
from enum import Enum
import inspect
from typing import Dict, Any, Callable, Optional


# 1. Explicit structure for an LLM action.
@dataclass
class AgentAction:
    tool_name: str
    args: Dict[str, Any]


def _result_metadata(value: Any) -> Dict[str, Any]:
    """Keep history useful without retaining tool output contents."""
    metadata = {"type": type(value).__name__}
    if isinstance(value, (str, bytes, bytearray, list, tuple, dict, set)):
        metadata["size"] = len(value)
    return metadata


class AgentRuntime:
    def __init__(
        self,
        fsm_engine,
        tools: Dict[str, Callable],
        max_steps: int = 5,
        decision_fn: Optional[Callable[[int], AgentAction]] = None,
        tool_validators: Optional[Dict[str, Callable[[Dict[str, Any]], None]]] = None,
    ):
        if not isinstance(tools, dict):
            raise TypeError("tools must be a dictionary")
        if any(not isinstance(name, str) or not name for name in tools):
            raise ValueError("tool names must be non-empty strings")
        if any(not callable(tool) for tool in tools.values()):
            raise TypeError("all tools must be callable")
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if decision_fn is not None and not callable(decision_fn):
            raise TypeError("decision_fn must be callable")
        tool_validators = tool_validators or {}
        if any(tool_name not in tools or not callable(validator)
               for tool_name, validator in tool_validators.items()):
            raise ValueError("tool validators must target callable registered tools")

        self.fsm = fsm_engine
        self.tools = tools
        self.max_steps = max_steps
        self.decision_fn = decision_fn or self._mock_llm_decide
        self.tool_validators = tool_validators
        self.tool_signatures = {}
        for tool_name, tool in tools.items():
            try:
                self.tool_signatures[tool_name] = inspect.signature(tool)
            except (TypeError, ValueError):
                self.tool_signatures[tool_name] = None
        self.history = []

    def _transition(self, state_name: str) -> None:
        """Transition using the FSM's enum type when one is exposed."""
        current_state = getattr(self.fsm, "state", None)
        target_state = state_name
        if isinstance(current_state, Enum):
            try:
                target_state = type(current_state)(state_name)
            except ValueError:
                target_state = state_name
        self.fsm.transition_to(target_state)

    def _validate_tool_args(self, action: AgentAction) -> None:
        if action.tool_name == "finish":
            if set(action.args) != {"result"} or not isinstance(action.args["result"], str):
                raise ValueError("finish requires only a string 'result' argument")
            return
        if action.tool_name not in self.tools:
            return

        signature = self.tool_signatures[action.tool_name]
        if signature is not None:
            try:
                signature.bind(**action.args)
            except TypeError as exc:
                raise ValueError(
                    f"Invalid arguments for tool '{action.tool_name}': {exc}"
                ) from exc

        validator = self.tool_validators.get(action.tool_name)
        if validator is not None:
            try:
                validator(action.args)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Tool argument validation failed for '{action.tool_name}'"
                ) from exc

    def _mock_llm_decide(self, step: int) -> AgentAction:
        """Simulate a probabilistic LLM decision engine."""
        if step == 1:
            return AgentAction(tool_name="read_file", args={"path": "data.txt"})
        elif step == 2:
            return AgentAction(tool_name="analyze_text", args={"content": "data_ok"})
        return AgentAction(tool_name="finish", args={"result": "Success"})

    def run(self, initial_input: str):
        if not isinstance(initial_input, str):
            raise TypeError("initial_input must be a string")
        print(f"[RUNTIME] Starting agent run for input_length={len(initial_input)}")
        
        for step in range(1, self.max_steps + 1):
            print(f"\n--- ReAct Step {step} ---")
            
            # 1. Receive a decision from the LLM.
            action = self.decision_fn(step)
            if not isinstance(action, AgentAction):
                raise TypeError("decision_fn must return an AgentAction")
            if not isinstance(action.tool_name, str) or not action.tool_name:
                raise ValueError("AgentAction.tool_name must be a non-empty string")
            if not isinstance(action.args, dict):
                raise TypeError("AgentAction.args must be a dictionary")
            print(
                f"[LLM INTENT] Proposed Tool: '{action.tool_name}' "
                f"with args metadata: {_result_metadata(action.args)}"
            )

            # 2. Check the guardrail and FSM state.
            if action.tool_name == "finish":
                self._validate_tool_args(action)
                self.fsm.transition_to("DONE")
                print("[RUNTIME] Task Completed Successfully.")
                return "DONE"

            # 3. Validate the tool name.
            if action.tool_name not in self.tools:
                print(f"[GUARD FAIL] Unknown tool: {action.tool_name}")
                continue

            self._validate_tool_args(action)

            # 4. Execute the tool in the activity layer.
            try:
                # Change the FSM state according to the action.
                if action.tool_name == "read_file":
                    self._transition("READING")
                elif action.tool_name == "analyze_text":
                    self._transition("ANALYZING")

                result = self.tools[action.tool_name](**action.args)
                print(f"[TOOL OUTPUT] Result metadata: {_result_metadata(result)}")
                
                # Record the result in short-term replay/context history.
                self.history.append({
                    "step": step,
                    "action": action.tool_name,
                    "result": _result_metadata(result),
                })

            except Exception as e:
                print(f"[RUNTIME ERROR] Tool execution failed: {type(e).__name__}")
                self._transition("FAILED")
                break

        print("[RUNTIME EXHAUSTED] Max steps reached without completion.")
        try:
            self._transition("FAILED")
        except (AttributeError, ValueError, PermissionError):
            pass
        return "FAILED"


# --- Runtime demo ---
if __name__ == "__main__":
    from types import SimpleNamespace

    # Simulate a simple FSM.
    fsm = SimpleNamespace(
        state="INIT"
    )

    def transition_to(state):
        fsm.state = state
        print(f"[FSM STATE CHANGE] -> {state}")

    fsm.transition_to = transition_to

    # Define deterministic tools allowed by the system.
    tool_registry = {
        "read_file": lambda path: f"Content of {path}",
        "analyze_text": lambda content: f"Analysis of {content}"
    }

    # Run the scenario.
    runtime = AgentRuntime(fsm_engine=fsm, tools=tool_registry)
    runtime.run("Analyze project files")
