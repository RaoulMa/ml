# -*- coding: utf-8 -*-
"""
Description: Python program for implementation of MergeSort.

Like QuickSort, Merge Sort is a Divide and Conquer algorithm. 
It divides input array in two halves, calls itself for the two halves 
and then merges the two sorted halves. The merge() function is used for 
merging two halves. The merge(arr, l, m, r) is key process that assumes 
that arr[l..m] and arr[m+1..r] are sorted and merges the two sorted 
sub-arrays into one.  

MergeSort(arr[], l,  r)
If r > l
     1. Find the middle point to divide the array into two halves:  
             middle m = (l+r)/2
     2. Call mergeSort for first half:   
             Call mergeSort(arr, l, m)
     3. Call mergeSort for second half:
             Call mergeSort(arr, m+1, r)
     4. Merge the two halves sorted in step 2 and 3:
             Call merge(arr, l, m, r)

"""

def merge(array, l, m, r):
    
    l_array = array[l:m]           # left arary
    r_array = array[m:r]           # right array
    
    i=0 # index for left array
    j=0 # index for right array
    k=l # index for sorted array
    
    while (i<len(l_array) and j<len(r_array)):
        if l_array[i]<=r_array[j]:
            array[k] = l_array[i]
            i+=1
        else:
            array[k] = r_array[j]
            j+=1
        k+=1
    
    while i<len(l_array):
        array[k] = l_array[i]
        i+=1
        k+=1
    
    while j<len(r_array):
        array[k] = r_array[j]
        j+=1
        k+=1
    

def merge_sort(array,l,r):
    if l<r:
        m = int((l+r)/2)
        merge_sort(array,l,m)
        merge_sort(array,m+1,r)
        merge(array,l,m,r)
        
def merge_sort2(alist):
    
    if len(alist)>1:
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        merge_sort2(lefthalf)
        merge_sort2(righthalf)

        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k]=lefthalf[i]
                i=i+1
            else:
                alist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            alist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            alist[k]=righthalf[j]
            j=j+1
            k=k+1
        

array = [12, 11, 13, 5, 6, 6, 7];
print("array: {}".format(array))

merge_sort(array,0,len(array))
print("sorted array: {}".format(array))
        
array = [12, 11, 13, 5, 6, 6, 7];
print("array: {}".format(array))

merge_sort2(array)
print("sorted array: {}".format(array))
        
    

