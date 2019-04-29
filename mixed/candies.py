#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alice is a kindergarten teacher. She wants to give some candies to 
the children in her class.  All the children sit in a line and each of 
them has a rating score according to his or her performance in the class.  
Alice wants to give at least 1 candy to each child. If two children sit next
 to each other, then the one with the higher rating must get more candies. 
 Alice wants to minimize the total number of candies she must buy.

For example, assume her students' ratings are [4, 6, 4, 5, 6, 2]. She 
gives the students candy in the following minimal amounts: [1, 2, 1, 2, 3, 1]. 
She must buy a minimum of 10 candies. 
"""

INF = 10**9 # a number larger than all ratings

# array
a = [9,8,6,1,1,8,6]
n = len(a)

# add sentinels
a = [INF] + a + [INF]

candies = [0]*(n+1)
# populate 'valleys'
for i in range(1,n+1):
    if a[i-1] >= a[i] <= a[i+1]:
        candies[i] = 1

# populate 'rises'
for i in range(1,n+1):
    if a[i-1] < a[i] <= a[i+1]:
        candies[i] = candies[i-1] + 1

# populate 'falls'
for i in range(n,0,-1):
    if a[i-1] >= a[i] > a[i+1]:
        candies[i] = candies[i+1] + 1

# populate 'peaks'
for i in range(1,n+1):
    if a[i-1] < a[i] > a[i+1]:
        candies[i] = max(candies[i-1], candies[i+1]) + 1

# print the total number of candies
print(sum(candies))