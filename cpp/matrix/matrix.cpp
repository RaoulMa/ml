#include "matrix.h"
#include <vector>
#include <iostream>
using namespace std;

Matrix::Matrix(){
    vector< vector<float> > init_grid(4, vector<float>(4,0));
    grid = init_grid;
    rows = grid.size();
    cols = grid[0].size();
}

Matrix::Matrix(vector< vector<float> > new_grid){
    grid = new_grid;
    rows = grid.size();
    cols = grid[0].size();
}

vector<float>::size_type Matrix::getRows(){
    return rows;
}

vector<float>::size_type Matrix::getCols(){
    return cols;
}

void Matrix::setGrid(vector< vector<float> > new_grid){
    grid = new_grid;
    rows = grid.size();
    cols = grid[0].size();
}

void Matrix::print(){

    for (int i=0;i<grid.size();i++){
        for (int j=0;j<grid[0].size();j++){
            cout << grid[i][j] << " ";
        }
        cout << endl;
    }

}

