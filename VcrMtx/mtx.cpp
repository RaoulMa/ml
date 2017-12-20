/*
Description: Implementing a template class for matrices.
Author: Raoul Malm
 */

#ifndef MTX_CPP //prevent multiple inclusion of mtx.cpp
#define MTX_CPP

#include <iostream>
using namespace std;

#include "vcr.cpp"

//abstract base class
template<class T> class AbsMtx {
protected:
public:
//	virtual Vcr<T> operator*(Vcr<T> &); //matrix vector multiplication
};

//full matrix class
template<class T> class FullMtx: public AbsMtx<T> {
private:
	int nr; //number of rows
	int nc; //number of columns
	T** mx; //entries of matrix (pointer)
public:
	FullMtx(int nr,int nc,T=0); //constructor
	FullMtx(int nr, int nc, T**); //constructor
	FullMtx(const FullMtx&); //copy constructor
	~FullMtx(){ //desctructor
		for(int i=0;i<this->nr;i++) delete[] mx[i];
		delete[] mx;
	}
	FullMtx& operator=(FullMtx&); //overload =
	FullMtx& operator+=(FullMtx&); //overload +=
	FullMtx& operator-=(FullMtx&); //overload -=
	FullMtx operator+(); //unary +
	FullMtx operator-(); //unary -
	FullMtx operator+(FullMtx&); //binary +
	FullMtx operator-(FullMtx&); //binary -
	T* operator[](int i) {return mx[i];} //subscript
	Vcr<T> operator*(Vcr<T>&); //matrix-vector multiplication
	void print();
};

//constructor
template<class T> FullMtx<T>::FullMtx(int nr, int nc, T a){
	this->nr=nr;
	this->nc=nc;
	mx = new T*[nr];
	for(int i=0;i<nr;i++){
		mx[i] = new T[nc];
		for (int j=0;j<nc;j++) mx[i][j] = a;
	}
}

//constructor
template<class T> FullMtx<T>::FullMtx(int nr, int nc, T** mxa){
	this->nr=nr;
	this->nc=nc;
	mx = new T*[nr];
	for(int i=0;i<nr;i++){
		mx[i] = new T[nc];
		for (int j=0;j<nc;j++) mx[i][j] = mxa[i][j];
	}
}

//overload =
template<class T> FullMtx<T>& FullMtx<T>::operator=(FullMtx& mat){
	if(this != &mat) {
		if (nr != mat.nr || nc != mat.nc)
			error("Matrix sizes do not match.");
		for (int i=0;i<nr;i++)
			for (int j=0;j<nc;j++)
				mx[i][j] = mat.mx[i][j];
	}
	return *this;
}

//overload +=
template<class T> FullMtx<T>& FullMtx<T>::operator+=(FullMtx& mat){
	if (nr != mat.nr || nc != mat.nc)
		error("Matrix sizes do not match.");
	for (int i=0;i<nr;i++)
			for (int j=0;j<nc;j++)
				mx[i][j] += mat.mx[i][j];
	return *this;
}

//overload -=
template<class T> FullMtx<T>& FullMtx<T>::operator-=(FullMtx& mat){
	if (nr != mat.nr || nc != mat.nc)
		error("Matrix sizes do not match.");
	for (int i=0;i<nr;i++)
			for (int j=0;j<nc;j++)
				mx[i][j] -= mat.mx[i][j];
	return *this;
}

//overload +
template<class T> FullMtx<T> FullMtx<T>::operator+(){
	return *this;
}

//overload -
template<class T> FullMtx<T> FullMtx<T>::operator-(){
	FullMtx<T> zero(nr,nc);
	zero -= *this;
	return zero;
}

//overload +
template<class T> FullMtx<T> FullMtx<T>::operator+(FullMtx& mat){
	FullMtx<T> sum(nr,nc);
	sum += *this;
	sum += mat;
	return sum;
}

//overload -
template<class T> FullMtx<T> FullMtx<T>::operator-(FullMtx& mat){
	FullMtx<T> diff(nr,nc);
	diff += *this;
	diff -= mat;
	return diff;
}

//print the matrix entries
template<class T> void FullMtx<T>::print(){
	for(int i=0;i<nr;i++){
		for(int j=0;j<nc;j++)
			cout << mx[i][j] << " ";
		if (i < nr-1) cout << ", ";
	}
}

//matrix-vector multiplication
template<class T> Vcr<T> FullMtx<T>::operator*(Vcr<T>& v) {
	if (nc != v.size())
		error("Matrix and vector size do not match.");
	Vcr<T> tm(nr);
	for (int i=0;i<nr;i++)
		for (int j=0;j<nc;j++) tm[i] += mx[i][j] * v[j];
	return tm;
}




#endif


