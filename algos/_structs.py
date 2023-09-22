from typing import Dict, List, Optional, Set, Tuple, Union


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Node:
    def __init__(self, val=0):
        self.val = val
        self.neighbors = []


def make_adj_list(n: int, edge_list: List[Tuple[int, int]]) -> List[List[int]]:
    """
    return an adjacency list for a graph with nodes labelled 0 through n-1 given a list of edges in the graph

    args:
    n: an integer representing the number of nodes in a graph
    edge_list: a list of tuples. Each tuple (u,v) represents an undirected edge.
    """
    adj_list: List[List[int]]= [[] for _ in range(n)]
    for edge in edge_list:
        u,v = edge
        adj_list[u].append(v)
        adj_list[v].append(u)
    return adj_list