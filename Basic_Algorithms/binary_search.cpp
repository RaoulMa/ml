/*
Description: Binary search
Author: Raoul Malm
*/

#include <iostream>
#include <vector>
using namespace std;

int compute_last_index_position_from_ordered_list(int num, vector<int> o_vector) {
    /*
     Compute last index position of a number appearing in ordered list.
     */
    int start=0;
    int end=o_vector.size()-1;
    
    while (start<=end){
        int mid = (start+end)/2;
        if (num==o_vector.at(mid) && (mid == end || o_vector.at(mid+1)>num)){
            return mid;
        }
        else if (num < o_vector.at(mid)){
            end = mid - 1;
        }
        else {
            start = mid + 1;
        }
    }
    
    return -1;
}


//main
int main() {

    cout << "-- Binary search --" << endl;
    
    int num = 3; // number
    int o_array[] = {0,1,2,3,3,3,4,5,6,7,8,9,10}; // array
    vector<int> o_vector(begin(o_array), end(o_array)); // create vector
    int idx = compute_last_index_position_from_ordered_list(num, o_vector);
    
    cout << "o_vector: ";
    for (int x = 0; x != o_vector.size(); ++x){
        cout << o_vector.at(x) << ",";
    }
    cout << endl << "num: " << num << endl;
    cout << "idx: " << idx << endl;
    
    
    
    
}

