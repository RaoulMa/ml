#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Given N and M find all stepping numbers in range N to M. A number 
is called as a stepping number if the adjacent digits have a difference of 1.
e.g 123 is stepping number, but 358 is not a stepping number
"""

def stepnum_1(A, B):

    step_num = []
    for n in range(A,B+1):
        n_str = str(n)
        c_prev = n_str[0]
        flag = True
        for i in range(1,len(n_str)):
            if abs(int(n_str[i])-int(c_prev)) != 1:
                flag = False
                break
            c_prev = n_str[i]
        if flag:
            step_num.append(n)
    return step_num

def stepnum_2(A, B):
    step_num = []
    for n in range(A,B+1):
        tmp = n
        d_prev = n%10
        flag = True
        while tmp>9:
            tmp = tmp//10
            d = tmp%10
            if abs(d_prev-d) != 1:
                flag = False
                break
            d_prev = d
        if flag:    
            step_num.append(n)
    return step_num


print(stepnum_1(10,100))
print(stepnum_2(10,100))