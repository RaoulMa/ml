
def fibonacci(n, cache={}):

    assert isinstance(n, int), 'error: Arg not of type int'
    assert n>=0, 'error: Arg smaller than zero'

    if n<=1:
        return 1
    if n not in cache:
        cache[n] = fibonacci(n-2) + fibonacci(n-1)
    return cache[n]


fibonacci(6)
print(type(2))