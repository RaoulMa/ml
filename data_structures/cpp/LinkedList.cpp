#include <iostream>
using namespace std;

class Node {
public:
    Node* next;
    int data;
    Node(int data);
};

class LinkedList {
public:
    Node* head;
    LinkedList();
    void print();
    void insert_at_head(int data);
    void insert_at_tail(int data);
    bool remove_at_key(int key);
};

Node::Node(int data){
    this->data = data;
    this->next = NULL;
}

LinkedList::LinkedList() {
    this->head = NULL;
}

void LinkedList::insert_at_head(int data) {
    Node* new_node = new Node(data);

    if (this->head != NULL) {
        new_node->next = this->head;
    }
    this->head = new_node;
}

void LinkedList::print(){
    Node* node = this->head;
    while (node != NULL){
        cout << node->data <<  " ";
        node = node->next;
    }
}

void LinkedList::insert_at_tail(int data) {

    Node* new_node = new Node(data);

    if (this->head == NULL) {
        this->head = new_node;
    }
    else {
        Node* node = this->head;
        while (node->next != NULL){
            node = node->next;
        }
        node->next = new_node;
    }
}

bool LinkedList::remove_at_key(int key) {

    if (this->head == NULL) {
        return false;
    }
    if (this->head->data == key) {
        this->head = this->head->next;
        return true;
    }
    Node* node = this->head;
    while (node->next != NULL){
        if (node->next->data == key) {
            node->next = node->next->next;
            return true;
        }
        node = node->next;
    }
    return false;
}

int main(){

    cout << "Linked List\n";

    LinkedList* list = new LinkedList();

    list->insert_at_tail(0);
    list->insert_at_head(1);
    list->insert_at_tail(2);
    list->insert_at_head(3);
    list->insert_at_head(4);

    list->print();

    list->remove_at_key(4);
    list->remove_at_key(0);
    list->remove_at_key(2);

    cout << "\n";
    list->print();

    cout << "\n";
    return 0;

}