#include "blur_factor_improved.hpp"

using namespace std;

vector < vector <float> > blur_factor_improved() {
    
    // calculate blur factors
    static float blurring = .12;
    static float center = 1.0 - blurring;
    static float corner = blurring / 12.0;
    static float adjacent = blurring / 6.0;
    
    // create the blur window
   static vector< vector<float> > window = {
        {corner, adjacent, corner},
        {adjacent, center, adjacent},
        {corner, adjacent, corner}
    };
    return window;
}

