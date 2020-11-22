def small_frog(X, A):
    # write your code in Python 3.6

    if X < 1:
        return -1

    if X > len(A):
        return -1

    set_ = set([n for n in range(1, X + 1)])

    for time, pos in enumerate(A):
        if pos in set_:
            set_.remove(pos)
        if len(set_) == 0:
            return time

    return -1

print(small_frog(5, [1, 3, 1, 4, 2, 3, 5, 4]))