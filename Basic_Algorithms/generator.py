#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Raoul Malm
Description: generator
"""

def integers():
    """Infinite sequence of integers."""
    i = 1
    while True:
        yield i
        i = i + 1

def squares():
    for i in integers():
        yield i * i

def take(n, seq):
    """Returns first n values from the given sequence."""
    result = []
    try:
        for i in range(n):
            result.append(seq.__next__())
    except StopIteration:
        pass
    return result

print(take(10, squares())) # prints [1, 4, 9, 16, 25]
g = squares()
print([next(g) for _ in range(10)])