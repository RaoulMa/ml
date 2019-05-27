class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList:

    def __init__(self):
        self.head = None

    def insert_at_beginning(self, value):
        new_node = Node(value)
        new_node.next = self.head
        self.head = new_node

    def __str__(self):
        values = ''
        node = self.head
        while node:
            values += str(node.value) + ' '
            node = node.next
        return values

    def insert_at_end(self, value):
        new_node = Node(value)
        node = self.head
        if not node:
            self.head = new_node
        else:
            while node.next:
                node = node.next
            node.next = new_node

    def remove(self, key):
        node = self.head
        prev_node = self.head

        if not node:
            return False

        if node.value == key:
            self.head = node.next
            return True

        while node:

            if node.value == key:
                prev_node.next = node.next
                return True
            else:
                prev_node = node
                node = node.next

        return False

linked_list = LinkedList()
linked_list.insert_at_end(0)
linked_list.insert_at_beginning(1)
linked_list.insert_at_beginning(2)
linked_list.insert_at_beginning(3)
linked_list.insert_at_beginning(4)
linked_list.insert_at_end(5)
linked_list.insert_at_end(6)
linked_list.remove(4)
linked_list.remove(0)

print(linked_list)






