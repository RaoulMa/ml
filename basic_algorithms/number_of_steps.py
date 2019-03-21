#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Davis has a number of staircases in his house and he likes to climb each 
staircase 1, 2 or 3 steps at a time. Being a very precocious child, he 
wonders how many ways there are to reach the top of the staircase.
"""

def num_steps(n, cache={}):
    count_steps = 0
    if n==0:
        return 1
    if n not in cache:
        for i in range(min(n,3)):
            count_steps += num_steps(n-(i+1))
        cache[n] = count_steps
    return cache[n]

print(num_steps(10))
