#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Given an array of integers, find the subset of non-adjacent elements
with the maximum sum. Calculate the sum of that subset. 
"""

def max_subset_sum(arr):
    if len(arr)==1:
        return arr
    if len(arr)==2:
        return max(arr[0],arr[1])
    
    max_minus_2 = max(0, arr[0])
    max_minus_1 = max(arr[1], max_minus_2)
    
    for n in range(2,len(arr)):
        max_n = max(max_minus_2, max_minus_2+arr[n], max_minus_1)
        max_minus_2 = max_minus_1
        max_minus_1 = max_n
        
    return max(max_minus_1, max_minus_2)

print(maxSubsetSum([1,-2,1,3,17,5]))
print(maxSubsetSum([3,5,-7,8,10]))