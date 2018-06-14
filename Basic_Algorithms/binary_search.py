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

if __name__ == "__main__":
    
    num = 3
    o_list = np.array([0,2,3,3,3,5,5,6,7,10])
    idx = compute_last_index_position_from_ordered_list(num, o_list)

    print('o_list: {}'.format(o_list))
    print('num: {}'.format(num))
    print('idx: {}'.format(idx))







