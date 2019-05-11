
class firstn_iterator():
    """ iterator """
    def __init__(self, n):
        self.n = n
        self.c = 0
    def __iter__(self):
        return self
    def __next__(self):
        self.c += 1
        if self.c > self.n:
            raise StopIteration()
        return self.c

def firstn_generator(n):
    """ generator """
    c = 0
    for _ in range(n):
        c += 1
        yield c

iterator = firstn_iterator(10)
print(*iterator)

generator = firstn_generator(10)
print(*generator)
