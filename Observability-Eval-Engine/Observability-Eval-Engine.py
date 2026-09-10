import time
from dataclasses import dataclass, field
from typing import List, Dict, Any


# 1. Span data structure for basic tracing.
@dataclass
class Span:
    name: str
    input_data: Any
    output_data: Any = None
    duration_ms: float = 0.0
    tokens_used: int = 0


_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "input",
    "password",
    "prompt",
    "query",
    "secret",
    "token",
    "output",
)


def _redact_value(key: str, value: Any) -> Any:
    """Prevent user content and credentials from remaining in local traces."""
    key_text = str(key).lower()
    if key_text in {"total_tokens", "prompt_tokens", "completion_tokens"} and isinstance(value, (int, float)):
        return value
    if any(part in key_text for part in _SENSITIVE_KEY_PARTS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(child_key): _redact_value(str(child_key), child_value) for child_key, child_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(key, child_value) for child_value in value]
    if isinstance(value, (bytes, bytearray)):
        return "[REDACTED]"
    if isinstance(value, str) and len(value) > 256:
        return f"{value[:256]}...[TRUNCATED]"
    return value


@dataclass
class Trace:
    trace_id: str
    spans: List[Span] = field(default_factory=list)

    def total_tokens(self) -> int:
        return sum(s.tokens_used for s in self.spans)


class AgentTracer:
    def __init__(self, trace_id: str):
        self.trace = Trace(trace_id=trace_id)

    def start_span(self, name: str, input_data: Any):
        return SpanTracker(self.trace, name, _redact_value("input_data", input_data))


class SpanTracker:
    def __init__(self, trace: Trace, name: str, input_data: Any):
        self.trace = trace
        self.span = Span(name=name, input_data=input_data)
        self.start_time = 0

    def __enter__(self):
        self.start_time = time.time()
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.span.duration_ms = (time.time() - self.start_time) * 1000
        self.span.input_data = _redact_value("input_data", self.span.input_data)
        self.span.output_data = _redact_value("output_data", self.span.output_data)
        self.trace.spans.append(self.span)


# 2. Evaluation harness.
class EvalEngine:
    @staticmethod
    def evaluate_run(trace: Trace, final_output: str) -> Dict[str, Any]:
        # Rule 1: Deterministic Check - Schema/Length
        length_ok = len(final_output) > 5

        # Rule 2: Token Budget Check
        budget_ok = trace.total_tokens() <= 500

        # Rule 3: deterministic demo signal; this is not a real LLM judge.
        normalized_output = final_output.strip().lower()
        faithfulness_score = 0.95 if normalized_output.startswith("valid ") else 0.40

        return {
            "passed": length_ok and budget_ok and (faithfulness_score > 0.7),
            "metrics": {
                "total_tokens": trace.total_tokens(),
                "spans_count": len(trace.spans),
                "faithfulness": faithfulness_score
            }
        }


# --- Performance demo ---
if __name__ == "__main__":
    tracer = AgentTracer(trace_id="run_9981")

    # Simulate one span, such as an LLM call.
    with tracer.start_span("LLM_Call", input_data="Summarize text") as span:
        time.sleep(0.05)  # Simulate latency.
        span.output_data = "Valid Summary Output"
        span.tokens_used = 120

    # Evaluate the output.
    results = EvalEngine.evaluate_run(tracer.trace, final_output="Valid Summary Output")
    print(f"[TRACE AUDIT] Total Tokens: {tracer.trace.total_tokens()} | Total Spans: {len(tracer.trace.spans)}")
    print(f"[EVAL RESULTS] Passed: {results['passed']} | Metrics: {results['metrics']}")
