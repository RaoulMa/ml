#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Bubble Sort
"""

def bubble_sort(list_):
    """Bubble sort algorithm.
    
    Args:
        list_ (list of ints): list of integers to be sorted
    
    """
    for n in range(len(list_)-1,0,-1):
        for i in range(n):
            if list_[i] > list_[i+1]:
                list_[i], list_[i+1] = list_[i+1], list_[i] 
                print(list_)

if __name__ == "__main__":
    
    list_ = [54,26,93,17,77,17,31,44,55,20]
    bubble_sort(list_)
    print(list_)