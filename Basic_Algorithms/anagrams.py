#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Checking anagrams
"""

def check_anagrams1(s1,s2):
    """Check anagrams.
    
    Run-time O(n^2)
    """
    
    if len(s1)!=len(s2):
        return False
    
    list_ = list(s2)
    
    for pos1 in range(len(s1)):
        
        found = False
        pos2 = 0
        
        while pos2 < len(list_) and found == False:
            
            if s1[pos1] == list_[pos2]:
                found = True
                list_[pos2] = None
            else:
                pos2 += 1

        if found == False:
            return False

    return True

def check_anagrams2(s1,s2):
    """Check anagrams
    
    Run-time: O(n log(n))
    """
    
    if len(s1)!=len(s2):
        return False
    
    list_1 = list(s1)
    list_2 = list(s2)

    list_1.sort()
    list_2.sort()

    for pos in range(0,len(s1)):
        if list_1[pos] != list_2[pos]:
            return False
    
    return True

def check_anagrams3(s1,s2):
    """Check anagrams.
    
    Run-time linear O(n)
    """
    if len(s1)!=len(s2):
        return False

    c1 = [0]*26
    c2 = [0]*26

    for i in range(len(s1)):
        pos = ord(s1[i])-ord('a')
        c1[pos] += 1
        
        pos = ord(s2[i])-ord('a')
        c2[pos] += 1

    for i in range(0,26): 
        if c1[i] != c2[i]:
            return False

    return True


str1 = "earth"
str2 = "heart"

print(str1,str2,check_anagrams1(str1,str2))
print(str1,str2,check_anagrams2(str1,str2))
print(str1,str2,check_anagrams3(str1,str2))