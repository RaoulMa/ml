#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In mathematics, the sieve of Eratosthenes is a simple, ancient algorithm 
for finding all prime numbers up to any given limit. 
"""

def find_primes(n):
    primes = [False,False]+[True]*(n-1)
    for i in range(2,int(n**0.5+1)):
        if primes[i] == True:
            for j in range(i*i,n+1,i):
                primes[j] = False
                
    return [i for i,x in enumerate(primes) if x]

print(find_primes(100))
    
    
    
