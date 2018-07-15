#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find peak elements in a given integer list
"""

def find_local_maxima(list_):
    """Find local maxima in given integer list.
    
    This is the brute force method.
    
    Args:
        list_ (list of ints): list of integers
    
    Returns:
        peak_elements (list of int): contains the local maxima elements
    
    """
    
    if not isinstance(list_, list):
        print('argument is not a list')
        return None
    
    n = len(list_)
    
    if n == 0:
        print('list is empty')
        return None
    
    if n == 1:
        return list_
    
    peak_elements = []
    
    if list_[0] > list_[1]:
        peak_elements.append(list_[0])
        
    if n > 3:
        for i in range(1,n-1):
            if list_[i-1] < list_[i] and list_[i] > list_[i+1]:
                peak_elements.append(list_[i])
    
    if n > 2 and list_[n-1] > list_[n-2]:
        peak_elements.append(list_[n-1])
    
    return peak_elements

def find_global_maximum(list_):
    """Find global maximum in unsorted list.
    
    Args:
        list_ (list of integers): list of integers
    
    Returns:
        max_ (int): maximum value
    
    """
    
    if not (isinstance(list_,list) or len(list_) < 1):
        print('argument not a list or empty')
        return None
    
    i = 0
    max_ = 0
    while i < len(list_):
        if list_[i] > max_:
            max_ = list_[i]
        i += 1
        
    return max_

def main():
    
    list_ = [7,3,6,2,8,1,2,7,2,1,0]
    print('given list: {}'.format(list_))
    print('local maxima: {}'.format(find_local_maxima(list_)))
    print('global maximum: {}'.format(find_global_maximum(list_)))
    
    
if __name__ == "__main__":
    main()

    

