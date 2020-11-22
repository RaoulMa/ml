def is_perm(A):
    # write your code in Python 3.6

    if len(A) == 0:
        return 0

    max_ = max(A)

    if len(A) != max_:
        return 0

    if len(set(A)) != max_:
        return 0

    return 1

print(is_perm([1,3,4]))
print(is_perm([1,2,4]))
print(is_perm([1,2,3]))
