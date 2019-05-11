def gen():
    for i in range(10):
        yield i

for value in gen():
    print(value)
