#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pseudo-random number generator
"""

def lcg(a,c,m,seed):
    """Linear congruential generator."""
    while True:
        seed = (a*seed+c)%m
        yield seed

g = lcg(3,1,17,1)
print([next(g) for _ in range(100)])




