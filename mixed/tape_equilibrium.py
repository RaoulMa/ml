def solution(A):
    # write your code in Python 3.6

    if len(A) < 2:
        return -1

    left_sum = A[0]
    right_sum = sum(A[1:])
    min_diff = abs(left_sum - right_sum)

    if len(A) == 2:
        return min_diff

    for n in range(1, len(A) - 1):
        left_sum += A[n]
        right_sum -= A[n]
        diff = abs(left_sum - right_sum)
        min_diff = min(min_diff, diff)

    return min_diff
