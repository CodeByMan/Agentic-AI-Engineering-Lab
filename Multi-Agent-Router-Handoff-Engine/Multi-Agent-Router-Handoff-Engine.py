import ast
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class AgentResponse:
    content: str
    next_agent: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


_FORBIDDEN_MODULES = {
    "builtins", "ctypes", "importlib", "marshal", "multiprocessing", "os",
    "pathlib", "pickle", "shutil", "socket", "subprocess", "sys",
}
_FORBIDDEN_NAMES = {
    "__builtins__", "__import__", "compile", "eval", "exec", "getattr",
    "globals", "open", "setattr", "vars",
}


def _value_metadata(value: Any) -> Dict[str, Any]:
    metadata = {"type": type(value).__name__}
    if isinstance(value, (str, bytes, bytearray, list, tuple, dict, set)):
        metadata["size"] = len(value)
    return metadata


def _find_unsafe_code_constructs(code: str) -> List[str]:
    violations = []
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in _FORBIDDEN_MODULES:
                    violations.append(f"forbidden module: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".", 1)[0] in _FORBIDDEN_MODULES:
                violations.append(f"forbidden module: {node.module}")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            violations.append(f"forbidden name: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            violations.append(f"forbidden attribute: .{node.attr}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_NAMES:
                violations.append(f"forbidden call: {node.func.id}()")
    return sorted(set(violations))


class Agent:
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt

    def execute(self, payload: Dict[str, Any]) -> AgentResponse:
        raise NotImplementedError


# --- Specialized agents ---

class TriageAgent(Agent):
    """Agent for initial task classification and routing."""
    def execute(self, payload: Dict[str, Any]) -> AgentResponse:
        task = payload.get("task", "")
        if not isinstance(task, str):
            raise TypeError("task must be a string")
        print(f"[{self.name}] Analyzing task metadata: {_value_metadata(task)}")
        
        if "code" in task.lower() or "bug" in task.lower():
            return AgentResponse(
                content="Task classified as Coding.",
                next_agent="CoderAgent",
                payload={"code_snippet": payload.get("task_data", "")}
            )
        return AgentResponse(content="Task unknown or completed.", next_agent=None)


class CoderAgent(Agent):
    """Specialized coding agent."""
    def execute(self, payload: Dict[str, Any]) -> AgentResponse:
        snippet = payload.get("code_snippet", "")
        if not isinstance(snippet, str):
            raise TypeError("code_snippet must be a string")
        print(f"[{self.name}] Fixing bug in snippet metadata: {_value_metadata(snippet)}")
        
        # Preserve the snippet as data inside a syntactically valid wrapper.
        fixed_code = f"def safe_func():\n    return {snippet!r}"
        return AgentResponse(
            content="Generated a syntactically valid literal wrapper.",
            next_agent="ReviewerAgent",
            payload={"fixed_code": fixed_code}
        )


class ReviewerAgent(Agent):
    """Agent for final review and approval."""
    def execute(self, payload: Dict[str, Any]) -> AgentResponse:
        fixed_code = payload.get("fixed_code", "")
        if not isinstance(fixed_code, str) or not fixed_code.strip():
            return AgentResponse(
                content="REJECTED: generated code must be a non-empty string",
                next_agent=None,
            )
        print(f"[{self.name}] Reviewing code metadata: {_value_metadata(fixed_code)}")
        
        try:
            violations = _find_unsafe_code_constructs(fixed_code)
        except (SyntaxError, TypeError) as exc:
            return AgentResponse(
                content=f"REJECTED: invalid generated code ({type(exc).__name__})",
                next_agent=None,
            )

        if violations:
            return AgentResponse(
                content=f"REJECTED: unsafe generated code ({'; '.join(violations)})",
                next_agent=None,
            )

        # Finish only after syntax validation.
        return AgentResponse(
            content=f"APPROVED: {fixed_code}",
            next_agent=None
        )


# --- Swarm orchestrator ---

class SwarmOrchestrator:
    def __init__(self, agents: List[Agent], max_handoffs: int = 5):
        self.agents: Dict[str, Agent] = {a.name: a for a in agents}
        self.max_handoffs = max_handoffs

    def run(self, start_agent_name: str, initial_payload: Dict[str, Any]) -> str:
        current_agent_name = start_agent_name
        current_payload = initial_payload
        handoff_history = []

        for step in range(self.max_handoffs):
            if current_agent_name not in self.agents:
                raise RuntimeError(f"Unknown agent: {current_agent_name}")

            # 1. Detect a short ping-pong handoff loop.
            if current_agent_name in handoff_history[-2:]:
                raise RuntimeError(f"[SWARM CRASH] Ping-pong loop detected on '{current_agent_name}'!")

            handoff_history.append(current_agent_name)
            agent = self.agents[current_agent_name]

            # 2. Execute the agent with an isolated payload.
            print(f"\n---> Step {step + 1}: Executing {current_agent_name}")
            response = agent.execute(current_payload)

            # 3. Check the handoff result.
            if not response.next_agent:
                print(f"[SWARM SUCCESS] Workflow terminated gracefully by {current_agent_name}.")
                return response.content

            # 4. Transfer control with only the returned payload.
            current_agent_name = response.next_agent
            current_payload = response.payload

        raise RuntimeError("[SWARM EXHAUSTED] Max handoffs budget exceeded.")


# --- Performance demo ---
if __name__ == "__main__":
    swarm_agents = [
        TriageAgent("TriageAgent", "Route tasks"),
        CoderAgent("CoderAgent", "Write code"),
        ReviewerAgent("ReviewerAgent", "Verify code")
    ]

    orchestrator = SwarmOrchestrator(agents=swarm_agents, max_handoffs=5)
    
    # Start the scenario.
    final_result = orchestrator.run(
        start_agent_name="TriageAgent",
        initial_payload={"task": "Fix a code bug", "task_data": "val = 1/0"}
    )
    print(f"\n[FINAL OUTPUT METADATA]: {_value_metadata(final_result)}")
