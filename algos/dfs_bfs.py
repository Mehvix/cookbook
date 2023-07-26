from collections import deque


from structs.tree import TreeNode

"""
BFS: queue (FIFO)
DFS: stack (LIFO)
"""

#*=======================================================================*#

def bfs_rows(root: TreeNode):
    # Base Case
    # if not root: return 0

    q = deque([root])

    while q:
        row_size = len(q)
        # row_sum = 0
        for _ in range(row_size):
            curr = q.popleft()  # deque is O(1) while list is O(n) [for any other elt than last]
            if curr.left:  q.append(curr.left)
            if curr.right: q.append(curr.right)
            # Work
            # row_sum += curr.val

    return

#*=======================================================================*#

out = 0
def dfs_backtrack(root: TreeNode):
    # Base Case
    # if not root: return 0

    def dfs(node):
        # if not node: return 0
        if not node.left and not node.right: return 0

        # Work; i.e. find diameter
        l=r=0
        if node.left:  l=1+dfs(node.left)
        if node.right: r=1+dfs(node.right)
        out = max(out, l+r)
        return max(l,r)

    dfs(root)

    return out


#*=======================================================================*#
