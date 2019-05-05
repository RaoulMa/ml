#include "initialize_matrix_improved.hpp"

using namespace std;
vector < vector<int> > initialize_matrix_improved(int num_rows, int num_cols, int initial_value) {
    vector < vector<int> > matrix(num_rows, vector<int>(num_cols, initial_value));
    return matrix;
}
