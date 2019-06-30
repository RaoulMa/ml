
def binary_gap(N):

    if N < 5:
        return 0

    start = 0
    gap_length = 0
    max_gap_length = 0

    for n in range(32):

        if N & 1 == 1:
            start = True
            max_gap_length = max(max_gap_length, gap_length)
            gap_length = 0

        elif star‚t:
            gap_length += 1

        N = N >> 1

    return max_gap_length

print(binary_gap(1))
print(binary_gap(5))
print(binary_gap(529))
