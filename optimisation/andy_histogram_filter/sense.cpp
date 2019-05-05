#include "headers/sense.h"

using namespace std;

vector< vector <float> > sense(char color, vector< vector <char> > &grid, vector< vector <float> > &beliefs,  float p_hit, float p_miss) 
{
	int i, j, height, width;
    height = grid.size();
	width = grid[0].size();
  	vector< vector <float> > newGrid(height, vector<float>(width, 0.0));

	for (i=0; i<height; i++) {
		for (j=0; j<width; j++) {
			if (grid[i][j] == color) {
				newGrid[i][j] = beliefs[i][j] * p_hit;
			}
            else {
				newGrid[i][j] = beliefs[i][j] * p_miss;
			}
		}
	}
	return newGrid;
}
