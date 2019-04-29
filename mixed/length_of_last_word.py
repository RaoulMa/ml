#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Given a string s consists of upper/lower-case alphabets and empty 
space characters ' ', return the length of last word in the string.
If the last word does not exist, return 0.
"""

def lengthOfLastWord(A):
    left = 0
    right = -1
    for n in range(0,len(A)):
        if n < len(A)-1 and A[n] == ' ' and A[n+1] != ' ':
            left = n+1
        elif A[n] != ' ':
            right = n                

    return right-left+1

print(lengthOfLastWord('Test  '))
print(lengthOfLastWord(' Test One Two  '))
print(lengthOfLastWord('     Hello'))
