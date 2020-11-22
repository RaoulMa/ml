
def passing_cars(A):

    if len(A) < 2:
        return 0

    sum_of_ones = sum(A)
    sum_of_passings = 0

    for pos, elem in enumerate(A):
        if elem == 0:
            sum_of_passings += sum_of_ones
        else:
            sum_of_ones -= 1

    return sum_of_passings

print(passing_cars( [0, 1, 0, 1, 1]))
