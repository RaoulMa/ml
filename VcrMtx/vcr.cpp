/*
Description: Implementing a template class for vectors.
Author: Raoul Malm
 */

#ifndef VCR_CPP //prevent multiple inclusion of vcr.cpp
#define VCR_CPP

#include <iostream>
using namespace std;

template<class T> class Vcr {
private:
	int len; //length of vector
	T* vr; //entries of vector (T pointer)
public:
	Vcr(int,int=0); //constructor
	Vcr(int, T*); //constructor
	Vcr(Vcr& v) {len=v.len; for(int i=0;i<len;i++) vr[i] = v.vr[i];}; //copy constructor
	~Vcr() {delete[] vr;} //destructor

	int size() const{return len;} //length of vector
	Vcr& operator = (Vcr&); //overload =
	Vcr& operator+=(Vcr&); // overload +=
	Vcr& operator-=(Vcr&); // overload -=
	Vcr operator+(); // overload +
	Vcr operator-(); // overload -
	Vcr operator+(Vcr&); // overload +
	Vcr operator-(Vcr&); // overload -
	T& operator[](int i){return vr[i];} //subscripting
	T maxnorm(); //maximum norm
	void print(); //print vector

	template<class S>
	friend S dot(Vcr<S>&,Vcr<S>&); //vector-vector multiplication
};

//error function
template<class T> void error(T& t){
	cout << t << " Program excited." << "\n";
	exit(1);
}

//constructor
template<class T> Vcr<T>::Vcr(int n,int a){
	len = n;
	vr = new T[len];
	for(int i=0;i<len;i++) vr[i]=a;
}

//constructor
template<class T> Vcr<T>::Vcr(int n, T* Tp){
	len = n;
	vr = new T[len];
	for(int i=0;i<len;i++) vr[i]=*(Tp+i);
}

//overload =
template<class T> Vcr<T>& Vcr<T>::operator=(Vcr<T>& v){
	if(this!=&v) {
		if(len != v.len) error("Vector sizes do not match.");
		for(int i=0;i<len;i++) vr[i] = v[i];
	}
	return *this;
}

//overload +=
template<class T> Vcr<T>& Vcr<T>::operator+=(Vcr<T>& v) {
	if(len != v.len) error("Vector sizes do not match.");
	for(int i=0;i<len;i++) vr[i] += v[i];
	return *this;
}

//overload -=
template<class T> Vcr<T>& Vcr<T>::operator-=(Vcr<T>& v) {
	if(len != v.len) error("Vector sizes do not match.");
	for(int i=0;i<len;i++) vr[i] -= v[i];
	return *this;
}

//overload +
template<class T> Vcr<T> Vcr<T>::operator+(){
	return *this;
}

//overload -
template<class T> Vcr<T> Vcr<T>::operator-(){
	Vcr<T> zero(len);
	zero -= *this;
	return zero;
}

//overload +
template<class T> Vcr<T> Vcr<T>::operator+(Vcr<T>& v){
	Vcr<T> sum(len);
	sum += *this;
	sum += v;
	return sum;
}

//overload -
template<class T> Vcr<T> Vcr<T>::operator-(Vcr<T>& v){
	Vcr<T> diff(len);
	diff += *this;
	diff -= v;
	return diff;
}

//maximum norm
template<class T> T Vcr<T>::maxnorm() {
	T nm = abs(vr[0]);
	for (int i=1;i<len;i++) nm = max(nm,abs(vr[i]));
	return nm;
}

//print vector
template<class T> void Vcr<T>::print() {
	for(int i=0;i<len;i++) cout << vr[i] << " ";
}

//dot product
template<class T> T dot(Vcr<T>& v1, Vcr<T>& v2){
	if (v1.len != v2.len) error("Vector sizes do not match.");
	T tm = v1[0]*v2[0];
	for (int i=1;i<v1.len;i++) tm += v1[i]*v2[i];
	return tm;
}


#endif


