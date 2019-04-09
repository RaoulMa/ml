#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
using namespace std;

class Matrix{

public:
    Matrix();
    Matrix(vector< vector<float> >);

    void print();
    void setGrid(vector< vector<float> >);
    vector<float>::size_type getRows();
    vector<float>::size_type getCols();

private:
    vector< vector<float> > grid;
    vector<float>::size_type rows;
    vector<float>::size_type cols;

};

#endif //MATRIX_H
