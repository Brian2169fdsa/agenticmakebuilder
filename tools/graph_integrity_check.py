"""
Graph Integrity Check

Performs structural analysis on the canonical spec's connection graph:
- DAG validation (no cycles)
- Orphan node detection
- Topological ordering
- Reachability analysis
- Self-loop detection

Input: trigger_id (int), module_ids (list[int]), connections (list[dict])
Output: dict with is_dag, orphan_nodes, topological_order, reachability, etc.

Deterministic. No network calls. No conversation context.
"""

from collections import defaultdict, deque


def graph_integrity_check(trigger_id, module_ids, connections):
    """Analyze the connection graph for structural integrity.

    Args:
        trigger_id: The trigger module ID (always 1).
        module_ids: List of all module IDs (excluding trigger).
        connections: List of connection dicts with 'from' and 'to' keys.
                     'from' can be "trigger" (string) or an integer module ID.

    Returns:
        dict with:
            - is_dag: bool — True if graph has no cycles
            - has_cycles: bool — True if cycles detected
            - cycle_nodes: list[int] — module IDs involved in cycles
            - orphan_nodes: list[int] — module IDs not reachable from trigger
            - self_loops: list[dict] — connections where from == to
            - topological_order: list — nodes in topological order (trigger first)
            - terminal_nodes: list[int] — nodes with no outgoing connections
            - reachability: dict — maps each node to set of reachable nodes
            - in_degree: dict — incoming edge count per node
            - out_degree: dict — outgoing edge count per node
    """
    all_nodes = {trigger_id} | set(module_ids)

    # Normalize "trigger" string to trigger_id integer
    def normalize_from(from_val):
        if from_val == "trigger":
            return trigger_id
        return from_val

    # Build adjacency list
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    self_loops = []

    for node in all_nodes:
        in_degree[node] = 0
        out_degree[node] = 0

    for conn in connections:
        src = normalize_from(conn["from"])
        dst = conn["to"]

        if src == dst:
            self_loops.append(conn)
            continue

        adj[src].append(dst)
        out_degree[src] += 1
        in_degree[dst] += 1

    # Cycle detection via DFS
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in all_nodes}
    cycle_nodes = set()

    def dfs_cycle(node):
        color[node] = GRAY
        for neighbor in adj[node]:
            if neighbor not in color:
                continue
            if color[neighbor] == GRAY:
                cycle_nodes.add(node)
                cycle_nodes.add(neighbor)
            elif color[neighbor] == WHITE:
                dfs_cycle(neighbor)
        color[node] = BLACK

    for node in all_nodes:
        if color[node] == WHITE:
            dfs_cycle(node)

    has_cycles = len(cycle_nodes) > 0

    # Topological sort via Kahn's algorithm
    topo_in = dict(in_degree)
    queue = deque()
    for node in all_nodes:
        if topo_in[node] == 0:
            queue.append(node)

    topological_order = []
    while queue:
        # Deterministic: always process lowest ID first
        queue = deque(sorted(queue))
        node = queue.popleft()
        topological_order.append(node)
        for neighbor in sorted(adj[node]):
            topo_in[neighbor] -= 1
            if topo_in[neighbor] == 0:
                queue.append(neighbor)

    # If topological order doesn't include all nodes, there are cycles
    if len(topological_order) != len(all_nodes):
        missing = all_nodes - set(topological_order)
        cycle_nodes.update(missing)
        has_cycles = True

    # Reachability from trigger via BFS
    reachable_from_trigger = set()

    def bfs_reachable(start):
        visited = set()
        q = deque([start])
        while q:
            node = q.popleft()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    q.append(neighbor)
        return visited

    reachable_from_trigger = bfs_reachable(trigger_id)
    orphan_nodes = sorted(all_nodes - reachable_from_trigger)

    # Full reachability map
    reachability = {}
    for node in sorted(all_nodes):
        reached = bfs_reachable(node)
        reached.discard(node)
        reachability[node] = sorted(reached)

    # Terminal nodes (no outgoing edges)
    terminal_nodes = sorted([n for n in all_nodes if out_degree[n] == 0])

    return {
        "is_dag": not has_cycles and len(self_loops) == 0,
        "has_cycles": has_cycles,
        "cycle_nodes": sorted(cycle_nodes),
        "orphan_nodes": orphan_nodes,
        "self_loops": self_loops,
        "topological_order": topological_order,
        "terminal_nodes": terminal_nodes,
        "reachability": reachability,
        "in_degree": dict(in_degree),
        "out_degree": dict(out_degree)
    }


def is_reachable(graph_result, from_id, to_id):
    """Check if to_id is reachable from from_id.

    Args:
        graph_result: Output from graph_integrity_check().
        from_id: Source node ID.
        to_id: Target node ID.

    Returns:
        bool
    """
    return to_id in graph_result["reachability"].get(from_id, [])


def get_predecessors(graph_result, node_id):
    """Get all nodes that can reach node_id.

    Args:
        graph_result: Output from graph_integrity_check().
        node_id: The target node.

    Returns:
        List of node IDs that have node_id in their reachability set.
    """
    return [
        nid for nid, reachable in graph_result["reachability"].items()
        if node_id in reachable
    ]


# --- Self-check ---
if __name__ == "__main__":
    print("=== Graph Integrity Check Self-Check ===\n")

    # Test 1: Simple linear DAG (trigger -> 2 -> 3 -> 4)
    print("Test 1: Linear DAG")
    result = graph_integrity_check(
        trigger_id=1,
        module_ids=[2, 3, 4],
        connections=[
            {"from": "trigger", "to": 2},
            {"from": 2, "to": 3},
            {"from": 3, "to": 4},
        ]
    )
    assert result["is_dag"] is True, "Linear graph should be a DAG"
    assert result["has_cycles"] is False
    assert result["orphan_nodes"] == []
    assert result["terminal_nodes"] == [4]
    assert result["topological_order"] == [1, 2, 3, 4]
    print("  [OK] Linear DAG validated")

    # Test 2: Graph with cycle (2 -> 3 -> 2)
    print("Test 2: Graph with cycle")
    result = graph_integrity_check(
        trigger_id=1,
        module_ids=[2, 3],
        connections=[
            {"from": "trigger", "to": 2},
            {"from": 2, "to": 3},
            {"from": 3, "to": 2},
        ]
    )
    assert result["is_dag"] is False, "Cyclic graph should not be a DAG"
    assert result["has_cycles"] is True
    assert 2 in result["cycle_nodes"] and 3 in result["cycle_nodes"]
    print("  [OK] Cycle detected")

    # Test 3: Graph with orphan
    print("Test 3: Graph with orphan")
    result = graph_integrity_check(
        trigger_id=1,
        module_ids=[2, 3, 4],
        connections=[
            {"from": "trigger", "to": 2},
            {"from": 2, "to": 3},
        ]
    )
    assert result["orphan_nodes"] == [4], f"Expected orphan [4], got {result['orphan_nodes']}"
    print("  [OK] Orphan node detected")

    # Test 4: Self-loop
    print("Test 4: Self-loop")
    result = graph_integrity_check(
        trigger_id=1,
        module_ids=[2],
        connections=[
            {"from": "trigger", "to": 2},
            {"from": 2, "to": 2},
        ]
    )
    assert result["is_dag"] is False, "Self-loop should not be a DAG"
    assert len(result["self_loops"]) == 1
    print("  [OK] Self-loop detected")

    # Test 5: Router with 2 branches (trigger -> 2(router) -> 3, 2 -> 4)
    print("Test 5: Router branching")
    result = graph_integrity_check(
        trigger_id=1,
        module_ids=[2, 3, 4],
        connections=[
            {"from": "trigger", "to": 2},
            {"from": 2, "to": 3},
            {"from": 2, "to": 4},
        ]
    )
    assert result["is_dag"] is True
    assert result["orphan_nodes"] == []
    assert result["out_degree"][2] == 2
    assert set(result["terminal_nodes"]) == {3, 4}
    print("  [OK] Router branching validated")

    # Test 6: Reachability helper
    print("Test 6: Reachability helper")
    result = graph_integrity_check(
        trigger_id=1,
        module_ids=[2, 3],
        connections=[
            {"from": "trigger", "to": 2},
            {"from": 2, "to": 3},
        ]
    )
    assert is_reachable(result, 1, 3) is True
    assert is_reachable(result, 3, 1) is False
    preds = get_predecessors(result, 3)
    assert 1 in preds and 2 in preds
    print("  [OK] Reachability helpers validated")

    print("\n=== All graph integrity checks passed ===")
