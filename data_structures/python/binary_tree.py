class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self, data):
        self.root = Node(data)

    def __str__(self):
        return str(self.preorder_traversal(self.root))

    def preorder_traversal(self, root, arr = []):
        if root:
            arr.append(root.data)
            self.preorder_traversal(root.left)
            self.preorder_traversal(root.right)
        return arr

    def search(self, value):
        return self.preorder_search(self.root, value)

    def preorder_search(self, root, value):
        if root:
            return root.data==value or self.preorder_search(root.left, value) \
                   or self.preorder_search(root.right, value)
        else:
            return False

tree = BinaryTree(1)
tree.root.left = Node(2)
tree.root.right = Node(3)
tree.root.left.left = Node(4)
tree.root.left.right = Node(5)
tree.root.right.left = Node(6)
tree.root.right.right = Node(7)

print(tree)

print(tree.search(6))
print(tree.search(8))


