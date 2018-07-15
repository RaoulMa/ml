#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: A palindromic number or numeral palindrome
is a number that remains the same when its digits are reversed.
"""

def palindromeNumbers(numlist): 
    """Check if given list contain palindrome numbers. 
    
    Python program to count and print all palindrome numbers in a list.
    
    Args: 
        numlist (list): List elements are checked if it contains a palindrome number
        
    Example:
        >>> palindromeNumbers([121, 133])
        121
        Total palindrome numbers are 1
    
    """
 
    c = 0
 
    # loop till list is not empty
    for n in numlist:             
        # Find reverse of current number
        t = n
        rev = 0
        while t > 0:
            rev = rev * 10 + t % 10
            t = int(t / 10)

        # compare rev with the current number
        if rev == n:
            print(n)
            c += 1
 
    print("Total palindrome numbers are", c)

class Deque:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def addFront(self, item):
        self.items.append(item)

    def addRear(self, item):
        self.items.insert(0,item)

    def removeFront(self):
        return self.items.pop()

    def removeRear(self):
        return self.items.pop(0)

    def size(self):
        return len(self.items)
    
def palindrome_checker(str_):
    chardeque = Deque()

    for ch in str_:
        chardeque.addRear(ch)

    while chardeque.size() > 1:
        first = chardeque.removeFront()
        last = chardeque.removeRear()
        if first != last:
            return False

    return True

def palindrome_checker_str(str_):
    
    str_new = ''
    for ch in str_:
        str_new = ch + str_new

    if str_new == str_:
        return True
        
    return False

 
def main():
    """Main function."""
 
    print('Palindrome checker using a deque data structure')
    print(palindrome_checker("lsdkjfskf"))
    print(palindrome_checker("radar"))

    
    print('\nPalindrome checker without using a deque')
    list_a = [10, 121, 133, 155, 141, 252]
    palindromeNumbers(list_a)
 
    list_b = [111, 220, 784, 565, 498, 787, 363]
    palindromeNumbers(list_b)   
    
    str_ = 'radar'
    print('{} is a palindrome: {}'.format(str_, palindrome_checker_str(str_)))

                      
 
if __name__=="__main__":
    main()             # main function call