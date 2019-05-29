import java.io.*;

class Node {
    int data;
    Node next=null;

    Node(int d) {
        data = d;
    }
}

public class LinkedList {
    Node head=null;

    public void insert_at_head(int data){
        Node new_node = new Node(data);
        new_node.next = head;
        head = new_node;
    }

    public void insert_at_tail(int data){
        Node new_node = new Node(data);

        if (head == null) {
            head = new_node;
        }
        else {
            Node node = head;
            while (node.next != null) {
                node = node.next;
            }
            node.next = new_node;
        }
    }

    public boolean remove_at_key(int key){

        if (head == null) {
            return false;
        }

        if (head.data == key) {
            head = head.next;
            return true;
        }

        Node node = head;
        while (node.next != null){
            if (node.next.data == key){
                node.next = node.next.next;
                return true;
            }
            node = node.next;
        }
        return false;
    }

    public void print(){
        Node node = head;
        while (node != null) {
            System.out.print(node.data + " ");
            node = node.next;
        }
    }

    public static void main(String args[]){

        System.out.println("Linked List");

        LinkedList list = new LinkedList();

        list.insert_at_head(0);
        list.insert_at_head(1);
        list.insert_at_tail(2);
        list.insert_at_head(3);
        list.insert_at_tail(4);
        list.print();

        System.out.println("");
        list.remove_at_key(4);
        list.remove_at_key(0);
        list.remove_at_key(2);
        list.remove_at_key(-1);
        list.print();

    }
}

