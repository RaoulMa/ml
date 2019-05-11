def root(n):
    """ root of n """
    x = n/3
    while True:
        x = 1/2 * (x+n/x)
        yield x

gen = root(4)
for _ in range(10):
    print(next(gen))


