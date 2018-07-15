#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: insertion sort
"""

def insertion_sort(list_):
    
   for idx in range(1,len(list_)):

     cur_value = list_[idx]
     pos = idx

     while pos > 0 and list_[pos-1] > cur_value:
         list_[pos] = list_[pos-1]
         pos -= 1

     list_[pos] = cur_value

list_ = [54,26,93,17,77,31,44,55,20]
insertion_sort(list_)
print(list_)