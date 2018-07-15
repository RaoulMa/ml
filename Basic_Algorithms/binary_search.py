# -*- coding: utf-8 -*-
"""
Description: Compute index position of given number in an (ascending) ordered list.
"""

import numpy as np
import math

def compute_last_index_position_from_ordered_list(num=3, o_list=np.arange(10)):
    ''' Compute last index position of a number appearing in ordered list'''
    
    start = 0 
    end = len(o_list)-1
    
    while (start <= end):
        mid = math.ceil((start+end)/2)
        
        if ((num == o_list[mid]) and (mid == end or o_list[mid+1] > num)):
            return mid
        elif num < o_list[mid]:
            end = mid-1
        else:
            start = mid +1
            
    return -1

def binary_search(item, list_):
    """Binary search without recursion."""
    
    low = 0
    high = len(list_)-1
      
    while low <= high:
        mid = (low+high)//2
        if list_[mid] == item:
            return True
        elif list_[mid] < item:
            low = mid+1
        else:
            high = mid-1
        
    return False        

def binary_search_with_recursion(item, list_):
    """Binary seach with recursion"""
    
    if len(list_) == 0:
        return False
    else:
        mid = len(list_)//2
        if list_[mid] == item:
            return True
        else:
            if list_[mid] > item:
                return binary_search_with_recursion(item, list_[:mid])
            else:
                return binary_search_with_recursion(item, list_[mid+1:])
    

if __name__ == "__main__":
    
    num = 3
    o_list = np.array([0,2,3,3,3,5,5,6,7,10])
    idx = compute_last_index_position_from_ordered_list(num, o_list)

    print('o_list: {}'.format(o_list))
    print('num: {}'.format(num))
    print('idx: {}'.format(idx))
    
    num = 6
    print('number {} was found: {}'.format(num, binary_search(num, o_list)))
    print('number {} was found: {}'.format(num, binary_search_with_recursion(num, o_list)))
    
    







