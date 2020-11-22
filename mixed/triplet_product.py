def triplet_product(A):
    # write your code in Python 3.6

    if len(A) < 3:
        return 0

    if len(A) == 3:
        return A[0] * A[1] * A[2]

    A.sort()

    max_ = A[-1] * A[-2] * A[-3]
    max_ = max(max_, A[-1] * A[0] * A[1])

    return max_

print(triplet_product())