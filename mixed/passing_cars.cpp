//  g++ -std=c++11 passing_cars.cpp && ./a.out

#include <iostream>
#include <vector>

using namespace std;

int passing_cars(vector<int> &A) {
    // write your code in C++14 (g++ 6.2.0)

    int sum_of_passings = 0;

    int sum_of_ones = 0;
    for (unsigned int i=0; i<A.size(); i++) {
        if (A[i] == 1) {
            sum_of_ones += 1;
        }
    }

    for (unsigned int i=0; i<A.size();i++) {
        if (A[i] == 0) {
            sum_of_passings += sum_of_ones;
        }
        else {
            sum_of_ones -= 1;
        }
    }

    return sum_of_passings;
}

int main(){
    vector<int> vec = {0, 1, 0, 1, 1};
    cout << passing_cars(vec);
}