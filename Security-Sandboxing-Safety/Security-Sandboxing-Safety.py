import ast
from typing import List, Tuple


MAX_SOURCE_LENGTH = 100_000


class SafeCodeASTChecker(ast.NodeVisitor):
    """Reject selected unsafe constructs in generated Python source."""

    FORBIDDEN_MODULES = {
        "builtins",
        "ctypes",
        "importlib",
        "marshal",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "shutil",
        "socket",
        "subprocess",
        "sys",
    }
    FORBIDDEN_NAMES = {
        "__builtins__",
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
    FORBIDDEN_ATTRIBUTES = {
        "call",
        "check_call",
        "check_output",
        "popen",
        "remove",
        "run",
        "system",
        "unlink",
    }

    def __init__(self, allowed_modules: List[str] = None):
        self.allowed_modules = set(allowed_modules or ["math", "json", "datetime", "re"])
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root_module = alias.name.split(".", 1)[0]
            if root_module in self.FORBIDDEN_MODULES:
                self.violations.append(f"Forbidden module import: '{alias.name}'")
            elif alias.name not in self.allowed_modules:
                self.violations.append(f"Forbidden module import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module_name = node.module or ""
        root_module = module_name.split(".", 1)[0]
        if root_module in self.FORBIDDEN_MODULES or module_name not in self.allowed_modules:
            self.violations.append(f"Forbidden module import: '{module_name}'")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in self.FORBIDDEN_NAMES:
            self.violations.append(f"Forbidden name: '{node.id}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith("_") or node.attr in self.FORBIDDEN_ATTRIBUTES:
            self.violations.append(f"Forbidden attribute access: '.{node.attr}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Reject direct calls to dynamic execution and file-access functions.
        if isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "open", "__import__", "compile"}:
                self.violations.append(f"Forbidden function call: '{node.func.id}()'")
        self.generic_visit(node)


def validate_agent_code(code_str: str, allowed_modules: List[str] = None) -> Tuple[bool, List[str]]:
    """Perform static policy validation before a separate execution environment."""
    if not isinstance(code_str, str):
        return False, ["Code must be provided as a string"]
    if len(code_str) > MAX_SOURCE_LENGTH:
        return False, [f"Code exceeds the {MAX_SOURCE_LENGTH}-character limit"]
    if allowed_modules is not None and not isinstance(allowed_modules, (list, tuple, set)):
        return False, ["allowed_modules must be a sequence of module names"]
    if allowed_modules is not None and any(
        not isinstance(module, str) or not module for module in allowed_modules
    ):
        return False, ["allowed_modules must contain non-empty strings"]

    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        return False, [f"Syntax Error: {type(exc).__name__}"]

    checker = SafeCodeASTChecker(allowed_modules=allowed_modules)
    checker.visit(tree)

    if checker.violations:
        return False, list(dict.fromkeys(checker.violations))
    return True, []


def execute_agent_code(*_args, **_kwargs):
    """Prevent callers from mistaking static validation for sandbox execution."""
    raise RuntimeError(
        "Untrusted code execution is disabled; use a separately isolated sandbox"
    )


# --- Security-policy demo ---
if __name__ == "__main__":
    # Code 1: an OS import that the policy must reject.
    malicious_code = """
import os
import math

def calculate(x):
    return math.sqrt(x)
"""

    # Code 2: a safe arithmetic example.
    safe_code = """
import math
import json

def process_data(val):
    result = math.pow(val, 2)
    return json.dumps({"result": result})
"""

    print("--- Checking Malicious Generated Code ---")
    is_safe, errors = validate_agent_code(malicious_code, allowed_modules=["math", "json"])
    print(f"Is Safe: {is_safe}")
    print(f"Violations: {errors}\n")

    print("--- Checking Safe Generated Code ---")
    is_safe, errors = validate_agent_code(safe_code, allowed_modules=["math", "json"])
    print(f"Is Safe: {is_safe}")
    print(f"Violations: {errors}")
