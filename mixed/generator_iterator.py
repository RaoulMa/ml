#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Generator and iterator
"""

# Building an iterator
class firstn(object):
    def __init__(self, n):
        self.n = n
        self.num, self.nums = 0, []

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        if self.num < self.n:
            cur, self.num = self.num, self.num+1
            return cur
        else:
            raise StopIteration()

# a generator that yields items instead of returning a list
def firstn(n):
    num = 0
    while num < n:
        yield num
        num += 1

sum_of_first_n = sum(firstn(1000000))

n = 1000000
sum_of_first_n = sum(firstn(n))
print(sum_of_first_n)
print(int(n*(n-1)/2))