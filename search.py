import heapq
from collections import defaultdict

def find_top_k_paths(start, end, k=5, graph=None, heuristic=None):
    """
    找到从start到end的前k条路径。
    graph: dict {node: [(neighbor, weight), ...]}
    heuristic: dict {node: heuristic_cost}
    """
    if graph is None:
        graph = {}  # 或抛异常，看你项目怎么设计

    if heuristic is None:
        heuristic = defaultdict(lambda: 0)

    queue = []
    heapq.heappush(queue, (0 + heuristic[start], 0, [start]))  # (f, g, path)
    found_paths = []
    visited_paths = set()

    while queue and len(found_paths) < k:
        f_score, g_score, path = heapq.heappop(queue)
        current = path[-1]

        if tuple(path) in visited_paths:
            continue
        visited_paths.add(tuple(path))

        if current == end:
            found_paths.append({'path': path, 'total_time': g_score})
            continue

        for neighbor, weight in graph.get(current, []):
            if neighbor in path:
                continue
            new_g = g_score + weight
            new_f = new_g + heuristic.get(neighbor, 0)
            heapq.heappush(queue, (new_f, new_g, path + [neighbor]))

    return found_paths
