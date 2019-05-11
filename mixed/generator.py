import time

def root(n):
    """ root of n """
    x = n/3
    while True:
        x = 1/2 * (x+n/x)
        yield x

def list_generator():
    for i in range(10000000):
        yield i

start = time.perf_counter()

_sum = 0
for value in list_generator():
    _sum += value

end = time.perf_counter()

print('value {} time {:.2f}'.format(_sum, end - start))

gen = root(4)
for _ in range(10):
    print(next(gen))

