from enum import StrEnum
from dataclasses import dataclass
import re
from typing import Callable, Dict, Any


class ModelTier(StrEnum):
    FAST_SLM = "FAST_SLM"       # Cheap and fast (for example, Llama-8B / Flash).
    HEAVY_LLM = "HEAVY_LLM"     # Expensive and deep reasoning (for example, Sonnet / GPT-4o).


@dataclass
class RouteDecision:
    selected_tier: ModelTier
    reason: str


class ModelRouter:
    """Route prompts using fast rules and input structure."""

    COMPLEX_KEYWORDS = frozenset({"proof", "refactor", "architect", "math", "analyze"})

    @staticmethod
    def route(prompt: str, requires_complex_json: bool = False) -> RouteDecision:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not isinstance(requires_complex_json, bool):
            raise TypeError("requires_complex_json must be a boolean")

        # 1. Invariants and rule-based checks.
        if requires_complex_json or len(prompt) > 2000:
            return RouteDecision(
                selected_tier=ModelTier.HEAVY_LLM,
                reason="High context length or complex schema requirement."
            )

        # 2. Heuristic check for complex-reasoning keywords.
        words = set(re.findall(r"[a-z0-9_]+", prompt.lower()))
        matched_keywords = sorted(words.intersection(ModelRouter.COMPLEX_KEYWORDS))

        if matched_keywords:
            return RouteDecision(
                selected_tier=ModelTier.HEAVY_LLM,
                reason=f"Complex reasoning keyword(s) detected: {', '.join(matched_keywords)}."
            )

        # 3. Default to the fast and inexpensive tier.
        return RouteDecision(
            selected_tier=ModelTier.FAST_SLM,
            reason="Simple task matched lightweight heuristic."
        )


class RouterExecutionEngine:
    """Execution engine with fallback cascading."""

    def __init__(self, providers: Dict[ModelTier, Callable]):
        if not isinstance(providers, dict):
            raise TypeError("providers must be a dictionary")
        if any(not callable(provider) for provider in providers.values()):
            raise TypeError("all model providers must be callable")
        self.providers = providers

    def execute_with_fallback(self, prompt: str, decision: RouteDecision) -> Any:
        if not isinstance(decision, RouteDecision):
            raise TypeError("decision must be a RouteDecision")
        primary_tier = decision.selected_tier
        fallback_tier = (
            ModelTier.HEAVY_LLM if primary_tier == ModelTier.FAST_SLM else ModelTier.FAST_SLM
        )

        if primary_tier not in self.providers:
            raise RuntimeError(f"No provider configured for primary tier: {primary_tier.value}")

        # First attempt: call the tier selected by the router.
        try:
            print(f"[ROUTER] Directing to '{primary_tier.value}' | Reason: {decision.reason}")
            return self.providers[primary_tier](prompt)
        except Exception as e:
            print(
                f"[ROUTER FALLBACK] '{primary_tier.value}' failed "
                f"({type(e).__name__}). Escalating to '{fallback_tier.value}'..."
            )
            # Second attempt: call the alternative tier.
            if fallback_tier not in self.providers:
                raise RuntimeError(
                    f"Primary tier failed and no fallback provider is configured: {fallback_tier.value}"
                ) from e
            return self.providers[fallback_tier](prompt)


# --- Performance demo ---
if __name__ == "__main__":
    # Simulate model providers.
    def mock_slm(p):
        if "fail" in p:
            raise RuntimeError("SLM Rate Limit Exceeded")
        return f"SLM Output for: {p}"

    def mock_heavy(p):
        return f"Heavy LLM Detailed Output for: {p}"

    providers_map = {
        ModelTier.FAST_SLM: mock_slm,
        ModelTier.HEAVY_LLM: mock_heavy
    }

    engine = RouterExecutionEngine(providers_map)

    # Scenario 1: simple request.
    d1 = ModelRouter.route("Hello, classify this text")
    print("Result 1:", engine.execute_with_fallback("Hello, classify this text", d1))
    print("-" * 50)

    # Scenario 2: complex request.
    d2 = ModelRouter.route("Analyze and refactor this architecture")
    print("Result 2:", engine.execute_with_fallback("Analyze and refactor this architecture", d2))
    print("-" * 50)

    # Scenario 3: fast-tier failure followed by fallback.
    d3 = ModelRouter.route("Simple text fail_test")
    print("Result 3:", engine.execute_with_fallback("Simple text fail_test", d3))
