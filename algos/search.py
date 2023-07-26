def binary_search(arr: list, target: int):
        n=len(arr)
        lo=0
        hi=n-1
        # Loop invariant: target is within arr[lo ... hi]
        while lo <= hi:
            mi = lo + (hi - lo) // 2
            if arr[mi] == target:
                return mi
            elif arr[mi] < target:
                lo=mi+1
            else:
                hi=mi-1

        return -1


def bst_2d(matrix, target):
        m=len(matrix)
        n=len(matrix[0])

        lo=0
        hi=m-1
        while lo <= hi:
            mi = lo + (hi - lo) // 2
            if target < matrix[mi][0]:
                hi=mi-1
            elif matrix[mi][-1] < target:
                lo=mi+1
            else:   # matrix[mi][0] <= target <= matrix[mi][n]
                break

        if not (lo <= hi): return False

        row=mi
        print(row)
        lo=0
        hi=n-1
        while lo <= hi:
            mi = lo + (hi - lo) // 2
            if matrix[row][mi] == target:
                return True
            elif matrix[row][mi] < target:
                lo=mi+1
            else:
                hi=mi-1

        return False