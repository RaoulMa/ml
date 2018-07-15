#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Selection Sort
"""

def selection_sort(list_):
    for n in range(len(alist)-1,0,-1):
        max_pos = 0
        for i in range(1,n+1):
            if alist[i] > alist[max_pos]:
                max_pos = i
    
        list_[i], list_[max_pos] = list_[max_pos], list_[i]
       
       
alist = [54,26,93,17,77,31,44,55,20]
selection_sort(alist)
print(alist)