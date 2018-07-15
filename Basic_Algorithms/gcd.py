#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Greates common divisor
"""

def gcd(m,n):
    while m%n != 0:
        t = m
        n = m%n
        m = t
    return n

def gcd_brute(m,n):
    i = n
    while (m%i != 0 or n%i != 0):
        i -= 1
    return i

def gcd_recursive(m,n):
    if n==0:
        return m
    else:
        return gcd_recursive(n, m%n)
    
    
    


a = 1071 
b = 462
g = gcd(a,b)
print("gcd(%i,%i) = %i"%(a,b,g))

g = gcd_brute(a,b)
print("gcd_brute(%i,%i) = %i"%(a,b,g))

g = gcd_recursive(a,b)
print("gcd_recursive(%i,%i) = %i"%(a,b,g))