/*
 The word polymorphism means having many forms. In programming, polymorphism means same function
 name (but different signatures) being uses for different types. Static polymorphism in Java is achieved
 by method overloading
 */

/*
Dynamic polymorphism in Java is achieved by method overriding. As the method to call is
determined at runtime, this is called dynamic binding or late binding.
*/

class Parent {
    public void print() {
        System.out.println("Parent");
    }
}

class Child1 extends Parent {
    public void print() {
        System.out.println("Child1");
    }
}

class Child2 extends Parent {
    public void print() {
        System.out.println("Child2");
    }
}

public class Polymorphism {

    public int sum(int x, int y){
        return x + y;
    }

    public int sum(int x, int y, int z){
        return x + y + z;
    }

    public static void main(String args[]){

        System.out.println("Static Polymorphism:");
        Polymorphism poly = new Polymorphism();
        System.out.println(poly.sum(1,2));
        System.out.println(poly.sum(1,2,3));

        System.out.println("Dynamic Polymorphism:");
        Parent obj = new Parent();
        obj.print();
        obj = new Child1();
        obj.print();
        obj = new Child2();
        obj.print();

    }
}




