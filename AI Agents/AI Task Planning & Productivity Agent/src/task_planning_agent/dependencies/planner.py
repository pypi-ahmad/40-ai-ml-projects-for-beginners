"""Dependency graph planner using NetworkX."""

from __future__ import annotations

import networkx as nx

from task_planning_agent.schemas import Task, TaskDependency


class DependencyPlanner:
    """Build and validate task dependency graph."""

    def build_dependencies(self, tasks: list[Task]) -> tuple[list[TaskDependency], list[str]]:
        id_to_task = {task.id: task for task in tasks}
        graph = nx.DiGraph()

        for task in tasks:
            graph.add_node(task.id)
            for dep in task.dependencies:
                if dep in id_to_task:
                    graph.add_edge(dep, task.id)

        issues: list[str] = []
        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            issues.extend([f"Dependency cycle detected: {' -> '.join(cycle)}" for cycle in cycles])

        topo_order = list(nx.topological_sort(graph)) if nx.is_directed_acyclic_graph(graph) else []
        order_map = {task_id: i for i, task_id in enumerate(topo_order)}

        dependencies: list[TaskDependency] = []
        for child in tasks:
            for parent_id in child.dependencies:
                if parent_id in id_to_task:
                    dependencies.append(
                        TaskDependency(
                            parent_task_id=parent_id,
                            child_task_id=child.id,
                            reason="Explicit dependency from extracted task metadata",
                        )
                    )
            child.reasoning = f"{child.reasoning} | topo_rank={order_map.get(child.id, -1)}"

        return dependencies, issues
