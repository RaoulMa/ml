/*
Description: Use Euler's Method to numerically solve simple first order
differential equations:
dx(t)/dt =  f(t,x(t)) ; x(t0) = x0; t0 < t < T 
Author: Raoul Malm
*/

#include <iostream>
#include <cmath>
using namespace std;

class ode {
private:
	double t0; //initial time
	double x0; //initial solution
	double T; //end time
	double (*fn)(double t, double x); //pointer to function
public:
	//constructor
	ode(double t0,double x0, double T, double (*fn)(double,double)) {
	this->t0 = t0; this->x0 = x0; this->T = T; this->fn = fn;
	}
	double* euler(int n) const; //explicit Euler's method
	double* eulerpc(int n) const; //pedictor-corrector Euler's method
	double* rk2(int n) const; //second-order Runge Kutta method
};

//definition of explicit Euler's method
double* ode::euler(int n) const {
	double* x = new double[n+1]; //x-array
	double h = (T-t0)/n; //step size
	x[0] = x0;
	for (int k=0;k<n;k++)
		x[k+1] = x[k] + h*fn(t0 + k*h,x[k]);
	return x;
}

//definition of predictor-corrector Euler's method
double* ode::eulerpc(int n) const {
	double* x = new double[n+1]; //x-array
	double h = (T-t0)/n; //step size
	x[0] = x0;
	for (int k=0;k<n;k++) {
		x[k+1] = x[k] + h*fn(t0 + k*h,x[k]);
		x[k+1] = x[k] + h*fn(t0 + (k+1)*h,x[k+1]);
	}
	return x;
}

//definition of second-order Runge Kutta method
double* ode::rk2(int n) const {
	double* x = new double[n+1]; //x-array
	double h = (T-t0)/n; //step size
	x[0] = x0;
	for (int k=0;k<n;k++){
		x[k+1] = x[k] + h*fn(t0 + k*h,x[k]);
		x[k+1] = 0.5*(x[k]+x[k+1] + h*fn(t0 + (k+1)*h,x[k+1]));
	}
	return x;
}

//example function
double fn(double t, double x) {
	return x*(1-exp(t))/(1+exp(t));
}

//exact solution to diff equation
double exact(double t) {
	return 12*exp(t)/pow(1+exp(t),2);
}

int main() {

	ode p(0,3,2,fn); //initialize object
	double* sol1 = p.euler(100); //explicit Euler's method
	double* sol2 = p.eulerpc(100); //predictor-corrector Euler's method
	double* sol3 = p.rk2(100); //second-order Runge Kutta method

	double norm1=0,norm2=0,norm3=0;
	double h=2.0/100;
	for (int k=1;k<=100;k++){
		norm1 = max(norm1,fabs(exact(k*h)-sol1[k]));
		norm2 = max(norm2,fabs(exact(k*h)-sol2[k]));
		norm3 = max(norm3,fabs(exact(k*h)-sol3[k]));
	}
	cout << "Error by explicit Euler's method = " << norm1 << '\n';
	cout << "Error by predictor-corrector Euler's method = " << norm2 << '\n';
	cout << "Error by second-order Runge Kutta method = " << norm3 << '\n';

}











