#include "headers/blur.h"

using namespace std;

vector < vector <float> > blur(vector < vector < float> > &grid, float blurring) {

	static float center = 1.0 - blurring;
	static float corner = blurring / 12.0;
	static float adjacent = blurring / 6.0;
	
  	vector < vector <float> > window = {{corner,adjacent,corner},
                                        {adjacent,center,adjacent},
                                        {corner,adjacent,corner}};
	vector<int> DX = {-1, 0, 1};
	vector<int> DY = {-1, 0, 1};

	static int height = grid.size();
	static int width = grid[0].size();
 	vector < vector <float> > newGrid(height, vector<float>(width, 0.0));
  
	float multiplier;
	int i, j, ii, jj, new_i, new_j, dx, dy;
	for (i=0; i< height; i++ ) {
		for (j=0; j<width; j++ ) {
			for (ii=0; ii<3; ii++) {
				dy = DY[ii];
	            new_i = (i + dy + height) % height;
				for (jj=0; jj<3; jj++) {
					dx = DX[jj];
					new_j = (j + dx + width) % width;
					newGrid[new_i][new_j] += grid[i][j] * window[ii][jj];
				}
			}
		}
	}

	return newGrid;
}
