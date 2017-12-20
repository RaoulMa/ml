/*
Description: Test vector and matrix template classes.
Author: Raoul Malm
 */

#include <iostream>
#include <cmath>
using namespace std;

#include "vcr.cpp"
#include "mtx.cpp"

int main() {

int n=3;
double* a = new double[n];
double* b = new double[n];
for (int i = 0;i<n;i++) {a[i]=-1.0*i; b[i] =1.5*i;}

Vcr<double> va(n,a);
Vcr<double> vb(n,b);

cout << "Test class Vcr";
cout << "\nVector a: "; va.print(); cout <<"Max Norm: " << va.maxnorm();
cout << "\nVector b: "; vb.print(); cout <<"Max Norm: " << vb.maxnorm();
cout << "\n-a: "; (-va).print();
cout << "\na+b: "; (va + vb).print();
cout << "\na-b: "; (va - vb).print();
cout << "\na*b: " << dot(va,vb);

double** A = new double*[n];
double** B = new double*[n];

for(int i=0;i<n;i++) {
	A[i] = new double[n];
	B[i] = new double[n];
	for(int j=0;j<n;j++) {
		A[i][j] = 1.0*(i+j)+i;
		B[i][j] = -1.0*(i-2*j)+i;
	}
}

FullMtx<double> MA(n,n,A);
FullMtx<double> MB(n,n,B);

cout << "\n\nTest class FullMtx";
cout << "\nMatrix A: "; MA.print();
cout << "\nMatrix B: "; MB.print();
cout << "\nA+B: "; (MA+MB).print();
cout << "\nA-B: "; (MA-MB).print();
cout << "\nA*a: "; (MA*va).print();




}
