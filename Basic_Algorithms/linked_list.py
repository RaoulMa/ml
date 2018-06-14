
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def insert_at_beginning(self, data):
        node = Node(data)
        node.next = self.head
        self.head = node
    
    def insert_at_end(self, data):
        val = self.head
        if val:  
            while(val.next):
                val = val.next
            val.next = Node(data)
        else:
            val = Node(data)
            
    def remove_node(self, key):

        cur = self.head
        prev = None
        
        while(cur):
            if cur.data == key:
                if cur.next and prev:
                    prev.next = cur.next
                elif prev:
                    prev.next = None
                else:
                    self.head = cur.next
                return
                
            if cur.next:
                prev = cur
                cur = cur.next
            else: 
                return
       
    def list_print(self):
        val = self.head
        while(val):
            print(val.data)
            val = val.next
        
linked_list = LinkedList()
linked_list.head = Node(12) 
linked_list.insert_at_beginning(17)
linked_list.insert_at_beginning(18)
linked_list.insert_at_beginning(21)
linked_list.insert_at_end(20)
linked_list.remove_node(17)


linked_list.list_print()


