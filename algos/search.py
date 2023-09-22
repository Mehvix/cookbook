from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np


def binary_search(arr: List[int], target: int):
    l,r = 0, len(arr)-1
    # Loop invariant: target is within arr[l ... r]
    while l <= r:
        m = l + (r - l) // 2
        if arr[m] == target:
            return m
        elif arr[m] < target:
            l = m + 1
        else:
            r = m - 1

    return -1

def bst_II(arr: List[int], target: int):
    # Use element's right neighbor to determine if the condition is met and decide whether to go left or right
    # Guarantees Search Space is at least 2 in size at each step
    l, r = 0, len(arr) - 1
    while l < r:
        m = l + (r - l) // 2
        if arr[m] == target:
            return m
        elif arr[m] < target:
            l = m + 1
        else:
            r = m

    # Post-processing:
    # End Condition: left == right
    if arr[l] == target:
        return l
    return -1

def bst_III(arr: List[int], target: int):
    # Use element's neighbors to determine if the condition is met and decide whether to go left or right
    # Guarantees Search Space is at least 3 in size at each step
    l, r = 0, len(arr) - 1
    while l + 1 < r:
        m = l + (r - l) // 2
        if arr[m] == target:
            return m
        elif arr[m] < target:
            l = m
        else:
            r = m

    # Post-processing:
    # End Condition: left + 1 == right
    if arr[l] == target: return l
    if arr[r] == target: return r
    return -1

# *=======================================================================* #

def lowest_indexOf(arr: List[int], target: int):
    """
    Implements an iterative version of binary search which returns the index of an element in an array.
    If there are multiple such elements, return the lowest index.

    args:
    lst: sorted lit of ints
    of: int which the function returns the index of
    """
    n = len(arr)
    l = 0
    r = n - 1

    while l <= r:
        mid = l + (r - l) // 2
        val_mid = arr[mid]
        if val_mid < target:
            l = mid + 1
        else:
            r = mid - 1
    l = max(0, min(l, n-1))
    return l if arr[l] == target else -1


def highest_indexOf(arr: List[int], target: int):
    n = len(arr)
    l = 0
    r = n - 1

    while l <= r:
        mid = l + (r - l) // 2
        val_mid = arr[mid]
        if val_mid <= target:
            l = mid + 1
        else:
            r = mid - 1

    r = max(0, min(r, n-1))
    return r if arr[r] == target else -1

# *=======================================================================* #

def bst_2d(matrix: List[List[int]], target: int):
    m = len(matrix)
    n = len(matrix[0])

    lo = 0
    hi = m - 1
    mi = -1
    while lo <= hi:
        mi = lo + (hi - lo) // 2
        if target < matrix[mi][0]:
            hi = mi - 1
        elif matrix[mi][-1] < target:
            lo = mi + 1
        else:  # matrix[mi][0] <= target <= matrix[mi][n]
            break

    if not (lo <= hi):
        return False

    assert mi != -1
    row = mi

    lo = 0
    hi = n - 1
    while lo <= hi:
        mi = lo + (hi - lo) // 2
        if matrix[row][mi] == target:
            return True
        elif matrix[row][mi] < target:
            lo = mi + 1
        else:
            hi = mi - 1

    return False

# *=======================================================================* #

def quick_select(arr: List[int], k: int, flip=False) -> int:
    """
    https://people.eecs.berkeley.edu/~vazirani/algorithms/chap2.pdf#page=11

    1. Randomly select a pivot element from the array
    2. Partion the array into three partitions (the elements less than, equal too, and greater than the pivot)
    3. Recurse on the partition which must contain the k-th smallest element
    """

    n = len(arr)
    pivot = arr[0] if flip else arr[n-1]     # alternate picking pivot from the front and back

    # partitions
    greather_than_pivot, equal_to_pivot, less_than_pivot = [],[],[]
    for e in arr:
        if e < pivot:
            greather_than_pivot.append(e)
        elif e == pivot:
            equal_to_pivot.append(e)
        else:
            less_than_pivot.append(e)


    # recurse on the partition which contains the k-th smallest element
    x, y = len(greather_than_pivot), len(equal_to_pivot)
    if k < x:
        return quick_select(greather_than_pivot, k, not flip)
    elif k < x + y:
        return pivot
    else:
        return quick_select(less_than_pivot, k - x - y, not flip)

# *=======================================================================* #

def longest_increasing_subsequence (arr: List[int], n: int):
    """
    https://people.eecs.berkeley.edu/~vazirani/algorithms/chap6.pdf#page=3

    Return a list containing longest increasing subsequence of the array.
    If there are ties, return any one of them.

    return: the longest increasing subsequence as a list -- return the actual
        elements, not the indices of the elements
    """
    last = [-1]*n
    count = [0]*n
    for i in range(n):
        for j in range(0,i):
            if arr[j]<arr[i]:
                if 1+count[j]>count[i]:
                    count[i] = 1+count[j]
                    last[i] = j

    l = np.argmax(count)
    order = [arr[l]]
    while last[l] != -1:
        order = [arr[last[l]]] + order
        l = last[l]

    return order