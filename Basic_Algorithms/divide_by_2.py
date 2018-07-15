#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Divide by 2 i.e. transform integer number of base 10 into binary string
"""

def base_converter(dec_number, base):
    """Convert a number in base 10 to any base.
    
    Args:
        dec_number (int): decimal number that should be converted
        base (int): new base
    
    Returns:
        str_ (str): representation of number in base
    
    """
    if not(isinstance(dec_number,int) or isinstance(base,int)):
        print('arguments have not the correct type: int')
        return None
    
    arr = []

    while dec_number > 0:
        rem = dec_number % base
        arr.append(rem)
        dec_number = dec_number // base

    str_ = ""
    for i in range(len(arr)-1,-1,-1):
        str_ = str_ + str(arr[i])

    return str_

def base_converter_recursive(n,base):
   convertString = "0123456789ABCDEF"
   if n < base:
      return convertString[n]
   else:
      return base_converter_recursive(n//base,base) + convertString[n%base]



if __name__ == "__main__":
    
    num = 233
    base = 2
    print('{} = {} in base {}'.format(num, base_converter(num,base), base))
    

    num = 233
    base = 2
    print('{} = {} in base {}'.format(num, base_converter(num,base), base))
    
    base = 5
    print('{} = {} in base {}'.format(num, base_converter(num,base), base))
    
    num = 25
    base = 8
    print('{} = {} in base {}'.format(num, base_converter(num,base), base))
    
    
    
