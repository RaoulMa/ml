#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check if a number is prime
"""

def check_primality(n):
    import math
    if n == 2:
        return True
    elif n == 1 or (n & 1) == 0:
        return False
        
    for i in range(2, math.ceil(math.sqrt(n)+1)):
        if (n % i) == 0:
            return False
        
    return True

print(check_primality(3))