#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the power set, i.e. all possible subsets of a set of integers.
"""

def get_subsets(arr):
    power_sets = [[]]

    if len(arr)==0:
        return power_sets
    
    for element in arr:
        len_ = len(power_sets)
        for j in range(len_):
            power_sets.append(power_sets[j]+[element])
    
    return power_sets
    
print(get_subsets([1,2,3]))