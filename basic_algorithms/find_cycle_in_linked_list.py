#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
"""

# Definition for singly-linked list.
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class LinkedList:
    def __init__(self, val):
        self.head = Node(val)

    def insert_at_beginning(self, val):
        node = Node(val)
        node.next = self.head
        self.head = node
       
    def list_print(self):
        node = self.head
        while(node):
            print(node.val)
            node = node.next

# @param A : head node of linked list
# @return the first node in the cycle in the linked list
def detect_cycle(A):
    count = set()
    node = A.head
    while node:
        if node.val in count:
            return node.val
        else:
            count.add(node.val)
            node = node.next
    return None

l_list = LinkedList(12)
l_list.insert_at_beginning(17)
l_list.insert_at_beginning(21)
l_list.insert_at_beginning(18)
l_list.insert_at_beginning(21)
l_list.insert_at_beginning(19)
l_list.list_print()

print('cycle point', detect_cycle(l_list))

