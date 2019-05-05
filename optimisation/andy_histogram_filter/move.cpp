#include "headers/move.h"
#include "headers/zeros.h"

using namespace std;

vector< vector <float> > move(int dy, int dx, 
	vector < vector <float> > &beliefs) 
{
	static int height = beliefs.size();
	static int width = beliefs[0].size();
	vector < vector <float> > newGrid;
	newGrid = zeros(height, width);

	int i, j, new_i, new_j;
  	for (i=0; i<height; i++) {
		new_i = (i + dy + height) % height;
		for (j=0; j<width; j++) {
			new_j = (j + dx + width)  % width;
			newGrid[new_i][new_j] = beliefs[i][j];
		}
	}
	return newGrid;
}
