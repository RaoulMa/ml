#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

int triplet_product(vector<int> &A) {
    // write your code in C++14 (g++ 6.2.0)

    if (A.size()<3) {
        return 0;
    }

    if (A.size()==3) {
        return A[0]*A[1]*A[2];
    }

    int start = 0;
    int end = A.size()-1;

    sort(A.begin(), A.end());

    int max1 = A[end]*A[end-1]*A[end-2];
    int max2 = A[end]*A[start]*A[start+1];

    if (max1>max2){
        return max1;
    }
    else {
        return max2;
    }

}

int main(){
    vector<int> vec = {-3,1,2,-2,5,6};

    cout << triplet_product(vec);

}