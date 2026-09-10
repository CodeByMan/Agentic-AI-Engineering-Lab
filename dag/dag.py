import asyncio
import inspect
from typing import Dict, List, Set, Any, Optional


class DAGNode:
    def __init__(self, name: str, action, dependencies: List[str] = None):
        self.name = name
        self.action = action
        self.dependencies = set(dependencies or [])


class DAGExecutor:
    def __init__(self, nodes: List[DAGNode], node_timeout: Optional[float] = None):
        if node_timeout is not None and node_timeout <= 0:
            raise ValueError("node_timeout must be greater than zero")

        self.nodes: Dict[str, DAGNode] = {}
        for node in nodes:
            if not node.name or not callable(node.action):
                raise ValueError("Each DAG node needs a name and callable action")
            if node.name in self.nodes:
                raise ValueError(f"Duplicate DAG node: {node.name}")
            self.nodes[node.name] = node

        missing_dependencies = {
            dependency
            for node in self.nodes.values()
            for dependency in node.dependencies
            if dependency not in self.nodes
        }
        if missing_dependencies:
            raise ValueError(f"Unknown DAG dependencies: {sorted(missing_dependencies)}")

        self._validate_acyclic()
        self.node_timeout = node_timeout

    def _validate_acyclic(self) -> None:
        remaining_dependencies = {
            name: set(node.dependencies) for name, node in self.nodes.items()
        }
        dependents: Dict[str, Set[str]] = {name: set() for name in self.nodes}
        for name, dependencies in remaining_dependencies.items():
            for dependency in dependencies:
                dependents[dependency].add(name)

        ready = [name for name, dependencies in remaining_dependencies.items() if not dependencies]
        visited = 0

        while ready:
            completed_name = ready.pop()
            visited += 1
            for dependent in dependents[completed_name]:
                remaining_dependencies[dependent].discard(completed_name)
                if not remaining_dependencies[dependent]:
                    ready.append(dependent)

        if visited != len(self.nodes):
            raise ValueError("Cycle detected in DAG")

    async def execute(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        completed: Set[str] = set()

        while len(completed) < len(self.nodes):
            # 1. Find pending nodes whose dependencies are complete.
            ready_nodes = [
                node for name, node in self.nodes.items()
                if name not in completed and node.dependencies.issubset(completed)
            ]

            if not ready_nodes:
                raise RuntimeError("Deadlock or Unresolved Cycle detected in DAG!")

            # 2. Execute all ready nodes in parallel (fan-out).
            print(f"[DAG] Executing in parallel: {[n.name for n in ready_nodes]}")
            tasks = [asyncio.create_task(self._run_node(node, results)) for node in ready_nodes]
            try:
                node_results = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

            # 3. Store results and mark the nodes complete.
            for name, res in node_results:
                results[name] = res
                completed.add(name)

        return results

    async def _run_node(self, node: DAGNode, current_results: Dict[str, Any]):
        # Inject completed dependency outputs as the node input.
        deps_data = {dep: current_results[dep] for dep in sorted(node.dependencies)}
        action_result = node.action(deps_data)
        if inspect.isawaitable(action_result):
            if self.node_timeout is None:
                res = await action_result
            else:
                res = await asyncio.wait_for(action_result, timeout=self.node_timeout)
        else:
            res = action_result
        return node.name, res


# --- Parallel-execution demo ---
async def main():
    async def fetch_web(ctx): 
        await asyncio.sleep(0.5)
        return "web_data"

    async def fetch_db(ctx): 
        await asyncio.sleep(0.5)
        return "db_data"

    async def summarize(ctx): 
        return f"Merged: {ctx['web']} + {ctx['db']}"

    nodes = [
        DAGNode("web", fetch_web),
        DAGNode("db", fetch_db),
        DAGNode("summary", summarize, dependencies=["web", "db"]),
    ]

    executor = DAGExecutor(nodes)
    output = await executor.execute()
    print("[FINAL OUTPUT]:", output["summary"])

if __name__ == "__main__":
    asyncio.run(main())
