
#include <iostream>

int binary_gap(int N) {
    // write your code in C++14 (g++ 6.2.0)

    if (N < 5) {
        return 0;
    }

    int start = 0;
    int gap_length = 0;
    int max_gap_length = 0;

    for (int n=0;n<32;n++) {

        if ((N & 1) == 1) {
            start = 1;
            max_gap_length = std::max(gap_length, max_gap_length);
            gap_length = 0;
        }
        else if (start == 1) {
            gap_length += 1;
        }

        N = (N >> 1);
    }

return max_gap_length;
}

int main() {

    std::cout << binary_gap(5) << '\n';
    std::cout << binary_gap(7) << '\n';
    std::cout << binary_gap(9) << '\n';

}