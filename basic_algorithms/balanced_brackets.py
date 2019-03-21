#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balanced brackets.
"""

def is_Balanced(s):
    table =  {')':'(', ']':'[', '}':'{'}
    stack = []
    for x in s:
        if stack and table.get(x) == stack[-1]:
            stack.pop()
        else:
            stack.append(x)
    
    print("NO" if stack else "YES")

is_Balanced('[({})]')