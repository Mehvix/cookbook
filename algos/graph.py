from heapq import heappop, heappush
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np


def djikstras(graph, k):
    """
    Find shortest path in a graph from a given vertex v to all other vertices.

    Arguments:
    - GRAPH is an adjacency dict representation of the undirected graph.
        graph[v] consists of tuples (u, d) such that (v, u) is an edge of weight d.
    - V is the start vertex from which we need to find the shortest distances.

    Return:
    - DISTANCE, a dictionary d such that d[u] is the length of the shortest path
        from V to u. By definition, d[V] = 0.
    - PARENT, a dictionary p such that p[u] is the parent of u on the shortest path
        from V to u. In other words, if the shortest path from V to u is (V, x, y, z, u),
        then p[u] = z, p[z] = y, ..., p[x] = V. We define p[V] to be None.
    """

    dist_node_PQ = []
    distance = {}
    parent = {}

    for u_edges in graph.values():
        for n, n_dist in u_edges:
            assert n_dist >= 0, "All edge weights must be positive."
            distance[n] = float("inf")
            parent[n] = None

    distance[k] = 0

    heappush(dist_node_PQ, (0, k))

    while len(dist_node_PQ):
        """
        In Python's implementation of a priority queue, it is difficult to change
        the key of an element that is already in the priority queue. This means that
        we cannot change a vertex's distance in the priority queue when the minimal
        distance to a vertex changes. For this reason, the priority queue might actually
        contain several entries for the same vertex. As we are taking out these
        entries we will check if they are "up-to-date". If the distance taken out of
        the priority queue is larger than the known minimal distance to this vertex,
        it means that this entry is out-of-date, so we can safely skip it.
        """
        u_dist, u = heappop(dist_node_PQ)
        if u_dist > distance[u]:
            continue

        for v, v_dist in graph[u]:
            v_new_dist = u_dist + v_dist
            if v_new_dist < distance[v]:
                distance[v] = v_new_dist
                parent[v] = u
                heappush(dist_node_PQ, (v_new_dist, v))

    assert parent[k] is None
    assert distance[k] == 0
    return distance, parent


def get_path(w, parent):
    """
    Take in a vertex W and a dictionary PARENT, and return the path from V to W,
    where V is the vertex from which we ran Dijkstra's algorithm.
    - PARENT, a dictionary p such that p[w] is the parent of w on the shortest path
        from V to w. In other words, if the shortest path from V to w is (V, x, y, z, w),
        then p[w] = z, p[z] = y, ..., p[x] = V. We define p[V] to be None.
    """
    out = []

    while w is not None:
        out = [w] + out
        w = parent[w]

    return out


def greatest_roads_solver(non_great_roads, greatest_roads, k, a, n):
    """
    Returns the shortest path which starts at node a and ends at node a which goes through k greatest roads

    args:
    non_great_roads: a list of tuples (u,v,d) containing all roads which are not greatest roads
    greatest_roads: a list of tuples containing all roads which are greatest roads
    k: an int representing the number of greatest roads the path must traverse
    a: the node representing home, which the path must start and end at
    n: the number of nodes in the graph
    """

    def get_path(a, parent, k, great):
        out = []
        while a is not None:
            out = [a] + out
            old_a = a
            a = parent[a][k]
            if (a, old_a) in great and k > 0:
                k -= 1

        return out

    all_roads = non_great_roads + greatest_roads

    adj_list_graph = {i: [] for i in range(n)}
    for u, v, d in all_roads:
        adj_list_graph[u].append((v, d))

    dist_node_layer_PQ = []
    distance = {}
    parent = {}

    for i in range(n):
        distance[i] = [float("inf")] * (k + 1)
        parent[i] = [None] * (k + 1)

    distance[a][0] = 0
    heappush(dist_node_layer_PQ, (0, a, 0))

    while len(dist_node_layer_PQ):
        u_dist, u, layer = heappop(dist_node_layer_PQ)
        if u_dist > distance[u][layer]:
            continue

        for v, v_dist in adj_list_graph[u]:
            lvl = layer + 1 if (u, v, v_dist) in greatest_roads and layer < k else layer
            v_new_dist = u_dist + v_dist
            if v_new_dist < distance[v][lvl]:
                distance[v][lvl] = v_new_dist
                parent[v][lvl] = u
                heappush(dist_node_layer_PQ, (v_new_dist, v, lvl))

    assert parent[a][0] is None
    assert distance[a][0] == 0
    return get_path(a, parent, k, [i[:2] for i in greatest_roads])


def longest_path_on_DAGS(adj_list, n, s, t):
    """
    https://people.eecs.berkeley.edu/~vazirani/algorithms/chap6.pdf

    Return a list containing the longest path on the dag. If there are ties, return
    any such path. If there are none, return the empty list.

    args:
    adj_list: an adjacency list representing the DAG, you can assume the topological
    ordering of vertices is simply their numeric order 0, 1, 2, 3, 4, ..., n-1
    n: number of nodes in the graph
    s: the source node of your path
    t: the target node in your path


    return: the longest path as a list of nodes the list [a, b, c, d, e] correspondes
    to the path a -> b -> c -> d -> e
    """
    if s == t:
        return [s]

    # reverse adjacency list
    rev_adj_list = [[] for _ in range(n)]
    for u, u_edges in enumerate(adj_list):
        if u >= s:
            for v, w in u_edges:
                if v >= s and v <= t and v != u:
                    rev_adj_list[v].append((u, w))

    last = [-1] * n
    count = [-float("inf")] * n
    count[s] = 0
    connected = [False] * n
    connected[s] = True
    for v in range(s, n):
        for u, w in rev_adj_list[v]:
            if connected[u]:
                connected[v] = True
                if w + count[u] > count[v]:
                    count[v] = w + count[u]
                    last[v] = u

    if not connected[t]:
        return []

    order = [t]
    while last[t] != -1 and last[t] != s:
        order = [last[t]] + order
        t = last[t]

    return [s] + order


def prims(points: List[List[int]]) -> int:
    n = len(points)
    adj = {i: [] for i in range(n)}  # (dist,pt)

    # for i,(x,y) in enumerate(points):
    #     for j,(a,b) in enumerate(points):
    #         if i!=j:
    #             d=abs(x-a)+abs(y-b)
    #             adj[i].append((d,j))
    #             adj[j].append((d,i))

    for i, (x1, y1) in enumerate(points):
        # x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dist = abs(x1 - x2) + abs(y1 - y2)  # Manhattan distance
            adj[i].append([dist, j])
            adj[j].append([dist, i])

    seen = set()
    heap: list[tuple[int, int]] = [(0, 0)]  # (cost,pt)
    out = 0
    while len(seen) < n:
        cost, u = heappop(heap)
        if u in seen:
            continue
        seen.add(u)
        out += cost
        for v_cost, v in adj[u]:
            if v not in seen:
                heappush(heap, (v_cost, v))

    return out
