#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Binary tree representation with nodes and references
"""

class BinaryTree:
    def __init__(self,key):
        self.key = key
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self,node):
        if self.leftChild == None:
            self.leftChild = BinaryTree(node)
        else:
            t = BinaryTree(node)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self,node):
        if self.rightChild == None:
            self.rightChild = BinaryTree(node)
        else:
            t = BinaryTree(node)
            t.rightChild = self.rightChild
            self.rightChild = t

    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self,key):
        self.key = key

    def getRootVal(self):
        return self.key

def preorder(tree):
    if tree:
        print(tree.getRootVal())
        preorder(tree.getLeftChild())
        preorder(tree.getRightChild())
        
def postorder(tree):
    if tree != None:
        postorder(tree.getLeftChild())
        postorder(tree.getRightChild())
        print(tree.getRootVal())
        
def inorder(tree):
  if tree != None:
      inorder(tree.getLeftChild())
      print(tree.getRootVal())
      inorder(tree.getRightChild())
        
tree = BinaryTree('a')
tree.insertLeft('b')
tree.insertRight('c')
tree.getLeftChild().insertLeft('d')
tree.getLeftChild().insertRight('e')
tree.getRightChild().insertLeft('f')
tree.getRightChild().insertRight('g')

print('preorder traversal:')
preorder(tree)

print('\ninorder traversal:')
inorder(tree)

print('\npostorder traversal:')
postorder(tree)
