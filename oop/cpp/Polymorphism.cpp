#include <iostream>
using namespace std;

/*
One of the key features of class inheritance is that a pointer to a derived class is
type-compatible with a pointer to its base class. Polymorphism is the art of taking advantage
of this simple but powerful and versatile feature.
*/

/*
A virtual member is a member function that can be redefined in a derived class,
while preserving its calling properties through references. The syntax for a function to
become virtual is to precede its declaration with the virtual keyword.
*/

class Parent {
protected:
    int a;
public:
    void set_a(int value) {this->a = value;}
    virtual void print() {cout << "Parent";}
};

class Child1: public Parent {
public:
    void print_a() {cout << this->a;}
    void print() {cout << "Child1";}
};

class Child2: public Parent {
public:
    void print_a() {cout << this->a;}
    void print() {cout << "Child2";}
};

int main(){

Child1 obj1;
Child2 obj2;

Parent* obj1_ptr = &obj1;
Parent* obj2_ptr = &obj2;

obj1_ptr -> set_a(1);
obj2_ptr -> set_a(2);

cout << "Polymorphism: type-compatible of a pointer\n";
obj1.print_a();
cout << " ";
obj2.print_a();
cout << "\n";

cout << "Polymorphism: virtual member functions \n";
obj1_ptr -> print();
cout << " ";
obj2_ptr -> print();
cout << "\n";


return 0;
}


