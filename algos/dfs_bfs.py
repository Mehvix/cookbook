from collections import deque

from algos._structs import Node, TreeNode

"""
BFS: queue (FIFO)
DFS: stack (LIFO)
"""

# *=======================================================================* #


def bfs_rows(root: TreeNode):
    # Base Case
    # if not root: return 0

    q = deque([root])

    while q:
        row_size = len(q)
        # row_sum = 0
        for _ in range(row_size):
            curr = q.popleft() # deque is O(1) while list is O(n) [for any other elt than last]
            if curr.left:
                q.append(curr.left)
            if curr.right:
                q.append(curr.right)
            # Work
            # row_sum += curr.val

    return


# *=======================================================================* #

def bfs_graph(root: Node, target):
    q = deque([root])
    seen = set([root])
    step = 0

    # BFS
    while q:
        size = len(q)
        for _ in range(size):
            cur = q.popleft()
            if cur == target:
                return step
            for next_node in cur.neighbors:
                if next_node not in seen:
                    q.append(next_node)
                    seen.add(next_node)

        step += 1

    return -1

# *=======================================================================* #

def dfs(root: Node, target):
    seen = set()
    stack = []

    stack.append(root)
    while stack:
        cur = stack.pop()
        if cur == target:
            return True
        if cur not in seen:
            seen.add(cur)
            for next_node in cur.neighbors:
                stack.append(next_node)

    return False

# *=======================================================================* #

def dfs_backtrack(root: TreeNode):
    # Base Case
    # if not root: return 0
    out = 0

    def dfs(node):
        nonlocal out
        
        # if not node: return 0
        if not node.left and not node.right:
            return 0

        # Work; i.e. find diameter
        l = r = 0
        if node.left:
            l = 1 + dfs(node.left)
        if node.right:
            r = 1 + dfs(node.right)
        out = max(out, l + r)
        return max(l, r)

    dfs(root)

    return out


# *=======================================================================* #
