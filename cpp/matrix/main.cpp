#include <iostream>
#include "matrix.h"
using namespace std;

int main(){

    cout << " Matrix Class \n";

    Matrix matrix;

    cout << matrix.getRows() << " rows and ";
    cout << matrix.getCols() << " columns \n";
    matrix.print();

}
