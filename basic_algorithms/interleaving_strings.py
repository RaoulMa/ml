#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.
"""

# @param A : string
# @param B : string
# @param C : string
# @return an integer
def isInterleave(A, B, C):
    
    if len(C) == 1:
        return (len(B) == 0 and C[0] == A[0]) or (len(A)==0 and C[0]==B[0])
    elif len(A)>0 and len(B)>0:
        return (A[0]==C[0] and isInterleave(A[1:],B[0:],C[1:])) or (B[0]==C[0] and isInterleave(A[0:],B[1:],C[1:]))
    elif len(A)==0 and len(B)>0:
        return (B[0]==C[0] and isInterleave(A[0:],B[1:],C[1:]))
    elif len(B)==0 and len(A)>0:
        return (A[0]==C[0] and isInterleave(A[1:],B[0:],C[1:]))
    else:
        False
    

print(isInterleave('B','ae','Bae'))

