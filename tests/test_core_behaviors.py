import ast
import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, module_alias: str | None = None):
    path = ROOT / relative_path
    module_name = module_alias or "test_module_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CoreBehaviorTests(unittest.TestCase):
    def test_security_rejects_indirect_dynamic_import(self):
        module = load_module("Security-Sandboxing-Safety/Security-Sandboxing-Safety.py")
        source = "__builtins__['__import__']('os').system('probe')"
        is_safe, violations = module.validate_agent_code(source)
        self.assertFalse(is_safe)
        self.assertTrue(violations)

    def test_rag_does_not_return_documents_for_unrelated_query(self):
        module = load_module("RAG-Context-Management/RAG-Context-Management.py")
        engine = module.HybridRAGEngine(["Pakistan architecture", "database indexing"])
        self.assertEqual(engine.hybrid_retrieve("not-in-corpus"), [])

    def test_retry_replays_cached_result_and_retries_transient_errors(self):
        module = load_module("Failure-Retry-Idempotency/Failure-Retry-Idempotency.py")
        attempts = []
        store = {}
        original_sleep = module.time.sleep
        module.time.sleep = lambda _: None
        try:
            def action(key):
                attempts.append(key)
                if len(attempts) < 3:
                    raise ConnectionError("temporary failure")
                return "done"

            self.assertEqual(
                module.execute_idempotent_activity(
                    "task", "step", action, idempotency_store=store
                ),
                "done",
            )
            self.assertEqual(
                module.execute_idempotent_activity(
                    "task", "step", action, idempotency_store=store
                ),
                "done",
            )
            self.assertEqual(len(attempts), 3)
        finally:
            module.time.sleep = original_sleep

    def test_retry_requires_scope_for_shared_store(self):
        module = load_module("Failure-Retry-Idempotency/Failure-Retry-Idempotency.py")
        with self.assertRaises(ValueError):
            module.execute_idempotent_activity("task", "step", lambda: "result")

        store = {}
        self.assertEqual(
            module.execute_idempotent_activity(
                "task", "step", lambda _key: "scope-a", idempotency_store=store
            ),
            "scope-a",
        )
        self.assertEqual(
            module.execute_idempotent_activity(
                "task", "step", lambda _key: "scope-b", idempotency_store={}, scope_id="scope-b"
            ),
            "scope-b",
        )

    def test_evaluator_rejects_invalid_substring_false_positive(self):
        module = load_module("Observability-Eval-Engine/Observability-Eval-Engine.py")
        result = module.EvalEngine.evaluate_run(module.Trace("test"), "This is an invalid answer")
        self.assertFalse(result["passed"])
        self.assertEqual(result["metrics"]["faithfulness"], 0.40)

    def test_trace_redacts_query_attributes(self):
        module = load_module("Observability-Tracing-Evaluation/Observability-Tracing-Evaluation.py")
        tracer = module.AgentTracer("test")
        span = tracer.start_span("query", "TOOL", {"query": "SELECT * FROM users", "total_tokens": 3})
        self.assertEqual(span.attributes["query"], "[REDACTED]")
        self.assertEqual(span.attributes["total_tokens"], 3)

    def test_eval_trace_redacts_input_and_output_data(self):
        module = load_module("Observability-Eval-Engine/Observability-Eval-Engine.py")
        tracer = module.AgentTracer("test")
        with tracer.start_span("llm", {"password": "secret"}) as span:
            span.output_data = {"token": "secret", "status": "ok"}
        self.assertEqual(tracer.trace.spans[0].input_data, "[REDACTED]")
        self.assertEqual(tracer.trace.spans[0].output_data, "[REDACTED]")

    def test_trace_redaction_covers_nested_attributes(self):
        module = load_module("Observability-Tracing-Evaluation/Observability-Tracing-Evaluation.py")
        tracer = module.AgentTracer("test")
        span = tracer.start_span(
            "request", "TOOL", {"metadata": {"password": "secret", "ok": "kept"}}
        )
        self.assertEqual(span.attributes["metadata"]["password"], "[REDACTED]")
        self.assertEqual(span.attributes["metadata"]["ok"], "kept")

    def test_coder_output_is_parseable_for_quoted_input(self):
        module = load_module("Multi-Agent-Router-Handoff-Engine/Multi-Agent-Router-Handoff-Engine.py")
        response = module.CoderAgent("Coder", "Write code").execute({"code_snippet": "x' + y"})
        fixed_code = response.payload["fixed_code"]
        ast.parse(fixed_code)
        review = module.ReviewerAgent("Reviewer", "Review code").execute(response.payload)
        self.assertTrue(review.content.startswith("APPROVED:"))

    def test_agent_runtime_rejects_unexpected_tool_arguments(self):
        module = load_module("agent-runtime/agent-runtime.py")

        class FakeFSM:
            def transition_to(self, _state):
                pass

        called = []
        runtime = module.AgentRuntime(
            FakeFSM(),
            {"read_file": lambda path: called.append(path) or "data"},
            decision_fn=lambda _step: module.AgentAction(
                "read_file", {"unexpected": "value"}
            ),
        )
        with self.assertRaises(ValueError):
            runtime.run("private task")
        self.assertEqual(called, [])

    def test_security_checker_rejects_oversized_code_and_execution(self):
        module = load_module("Security-Sandboxing-Safety/Security-Sandboxing-Safety.py")
        is_safe, violations = module.validate_agent_code("x" * (module.MAX_SOURCE_LENGTH + 1))
        self.assertFalse(is_safe)
        self.assertTrue(violations)
        is_safe, violations = module.validate_agent_code("def broken(:\n    pass")
        self.assertFalse(is_safe)
        self.assertNotIn("broken", " ".join(violations))
        with self.assertRaises(RuntimeError):
            module.execute_agent_code("print('blocked')")

    def test_reviewer_rejects_unsafe_generated_code(self):
        module = load_module("Multi-Agent-Router-Handoff-Engine/Multi-Agent-Router-Handoff-Engine.py")
        response = module.ReviewerAgent("Reviewer", "Review code").execute(
            {"fixed_code": "import os\nos.system('whoami')"}
        )
        self.assertTrue(response.content.startswith("REJECTED: unsafe"))

    def test_durable_activity_replays_by_step_id(self):
        module = load_module("durable_execution/durable_execution.py")
        with tempfile.TemporaryDirectory() as directory:
            original_log = module.LOG_FILE
            module.LOG_FILE = Path(directory) / "execution_log.json"
            calls = []
            try:
                self.assertEqual(
                    module.run_activity("step", lambda: calls.append(1) or "result"),
                    "result",
                )
                self.assertEqual(
                    module.run_activity("step", lambda: calls.append(2) or "other"),
                    "result",
                )
                self.assertEqual(calls, [1])
            finally:
                module.LOG_FILE = original_log

    def test_durable_activity_scopes_replay_by_run_id(self):
        module = load_module("durable_execution/durable_execution.py")
        with tempfile.TemporaryDirectory() as directory:
            original_log = module.LOG_FILE
            module.LOG_FILE = Path(directory) / "execution_log.json"
            calls = []
            try:
                action = lambda: calls.append(1) or "result"
                self.assertEqual(module.run_activity("step", action, run_id="run-a"), "result")
                self.assertEqual(module.run_activity("step", action, run_id="run-b"), "result")
                self.assertEqual(calls, [1, 1])
                payload = json.loads(module.LOG_FILE.read_text(encoding="utf-8"))
                self.assertEqual(set(payload["runs"]), {"run-a", "run-b"})
            finally:
                module.LOG_FILE = original_log

    def test_durable_activity_replays_legacy_flat_log(self):
        module = load_module("durable_execution/durable_execution.py")
        with tempfile.TemporaryDirectory() as directory:
            original_log = module.LOG_FILE
            module.LOG_FILE = Path(directory) / "execution_log.json"
            module.LOG_FILE.write_text(
                json.dumps({"legacy-step": "legacy-result"}), encoding="utf-8"
            )
            try:
                self.assertEqual(
                    module.run_activity("legacy-step", lambda: "new-result"),
                    "legacy-result",
                )
            finally:
                module.LOG_FILE = original_log

    def test_durable_and_fsm_reject_malformed_state(self):
        durable = load_module("durable_execution/durable_execution.py")
        fsm_module = load_module("fsm/fsm.py")
        with tempfile.TemporaryDirectory() as directory:
            durable_log = Path(directory) / "execution_log.json"
            durable_log.write_text("{", encoding="utf-8")
            original_log = durable.LOG_FILE
            durable.LOG_FILE = durable_log
            try:
                with self.assertRaises(RuntimeError):
                    durable.run_activity("step", lambda: "result")
            finally:
                durable.LOG_FILE = original_log

            checkpoint = Path(directory) / "checkpoint.json"
            checkpoint.write_text("{", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                fsm_module.PersistentFSM(str(checkpoint))

    def test_dag_validates_dependencies_cycles_and_timeout(self):
        module = load_module("dag/dag.py")

        async def action(_ctx):
            return "ok"

        with self.assertRaises(ValueError):
            module.DAGExecutor([module.DAGNode("a", action, ["missing"])])
        with self.assertRaises(ValueError):
            module.DAGExecutor(
                [
                    module.DAGNode("a", action, ["b"]),
                    module.DAGNode("b", action, ["a"]),
                ]
            )

        async def slow(_ctx):
            await asyncio.sleep(0.1)
            return "late"

        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(
                module.DAGExecutor(
                    [module.DAGNode("slow", slow)], node_timeout=0.001
                ).execute()
            )

        def sync_action(_ctx):
            return "sync-ok"

        result = asyncio.run(
            module.DAGExecutor([module.DAGNode("sync", sync_action)]).execute()
        )
        self.assertEqual(result["sync"], "sync-ok")

    def test_agent_runtime_converts_enum_fsm_states(self):
        fsm_module = load_module("fsm/fsm.py", "fsm_enum")
        runtime_module = load_module("agent-runtime/agent-runtime.py", "runtime_enum")
        with tempfile.TemporaryDirectory() as directory:
            fsm = fsm_module.PersistentFSM(str(Path(directory) / "checkpoint.json"))
            runtime = runtime_module.AgentRuntime(
                fsm,
                {"read_file": lambda path: "data"},
                decision_fn=lambda _step: runtime_module.AgentAction(
                    "read_file", {"path": "data.txt"}
                ),
                max_steps=1,
            )
            self.assertEqual(runtime.run("read"), "FAILED")
            self.assertEqual(fsm.state, fsm_module.State.FAILED)

    def test_memory_persists_metadata_and_validates_input(self):
        module = load_module("Memory-Systems/Memory-Systems.py")
        with tempfile.TemporaryDirectory() as directory:
            storage_path = Path(directory) / "memory.json"
            store = module.LongTermMemoryStore(str(storage_path))
            store.upsert_fact("User", "Location", "Karachi", confidence=0.9)
            reloaded = module.LongTermMemoryStore(str(storage_path))
            metadata = reloaded.query_entity("user", include_metadata=True)
            self.assertEqual(metadata["location"]["value"], "Karachi")
            self.assertEqual(metadata["location"]["source"], "conversation")
            self.assertEqual(metadata["location"]["confidence"], 0.9)

            with self.assertRaises(ValueError):
                store.upsert_fact("", "location", "Karachi")

    def test_model_router_uses_word_boundaries_and_requires_fallback(self):
        module = load_module("model-routing/model-routing.py")
        self.assertEqual(
            module.ModelRouter.route("This is an analysis").selected_tier,
            module.ModelTier.FAST_SLM,
        )
        self.assertEqual(
            module.ModelRouter.route("Please analyze this").selected_tier,
            module.ModelTier.HEAVY_LLM,
        )

        def failing_provider(_prompt):
            raise RuntimeError("provider unavailable")

        engine = module.RouterExecutionEngine({module.ModelTier.FAST_SLM: failing_provider})
        decision = module.ModelRouter.route("simple request")
        with self.assertRaises(RuntimeError):
            engine.execute_with_fallback("simple request", decision)


if __name__ == "__main__":
    unittest.main()
