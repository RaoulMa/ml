/*
Description: Basic example that demonstrates the run-time polymorphism
(dynamic binding). The idea is that the correspondence between a function
and the type of an object is established at run-time. If the connection can
be established at compiler-time one speaks of static polymorphism.
Author: Raoul Malm
*/

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

//1d point class
class Pt {
private:
	double x;
public:
	Pt(double a = 0) {x = a;}
	/* declare draw() as virtual such that the correct correspondence
	   between the type and the different versions of draw() can be established
	   at run-time */
	virtual void draw() const {cout<<x;}
};

//2d point class
class Pt2d: public Pt {
private:
	double y;
public:
	Pt2d(double a=0,double b=0): Pt(a), y(b) {}
	void draw() const {
		Pt::draw();
		cout << " " << y;
	}
};

//3d point class
class Pt3d: public Pt2d {
private:
	double z;
public:
	Pt3d(double a = 0, double b=0,double c=0): Pt2d(a,b), z(c) {}
	void draw() const{
		Pt2d::draw();
		cout << " " << z;
	}
};

//global function
void h(const vector<Pt*>& v) {
	for(int i=0;i<v.size();i++){
		/* whether v[i] corresponds to Pt, Pt2d, Pt3d can only be known
		 * at run-time -> dynamic binding */
		v[i]->draw();
		cout << '\n';
	}
}

template <typename T>
T max_sub_array(vector<T> const & numbers){
	T max_ending = 0, max_so_far = 0;

	return max_so_far;
}

//main
int main() {
Pt a(5);
Pt2d b(4,9);
Pt3d c(7,7,7);

vector<Pt*> v(3); //vector of type Pt* with 3 components
v[0] = &a;
v[1] = &b;
v[2] = &c;
h(v);

}

