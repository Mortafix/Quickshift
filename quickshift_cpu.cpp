#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "quickshift_cmn.h"
#include "common.h"

// distance between data at pixel i and j along K channels and adding the distance between i and j
float distance(float const * data, int height, int width, int channels, int x_col, int x_row, int y_col, int y_row){
	int d1 = y_col - x_col;
	int d2 = y_row - x_row;
	float dist = d1 * d1 + d2 * d2;
	for (int k = 0; k < channels; k++) {
		float d = data[x_col + height * x_row + (height*width) * k] - data[y_col + height * y_row + (height*width) * k];
		dist += d * d;
	}
	return dist;
}

void quickshift_cpu(qs_image image, float sigma, float dist, float * map, float * gaps, float * E, float * time){

	// variables
	float const * data = image.data;
	int height = image.height;
	int width = image.width;
	int channels = image.channels;
	int R = (int) ceil (3 * sigma);
	int Rd = (int) ceil (dist);

	double start = seconds();
	
	// for every pixel in the image compute its density
	for (int x_row = 0; x_row < width; x_row++) {
		for (int x_col = 0; x_col < height; x_col++) {

			// initialize boundaries from sigma
			float Ei = 0;
			int y_col_min = MAX(x_col - R, 0	 );
			int y_col_max = MIN(x_col + R, height-1);
			int y_row_min = MAX(x_row - R, 0	 );
			int y_row_max = MIN(x_row + R, width-1);

			// for each pixel in the area (sigma) compute the distance between it and the source pixel
			for (int y_row = y_row_min; y_row <= y_row_max; ++ y_row) {
				for (int y_col = y_col_min; y_col <= y_col_max; ++ y_col) {
					float Dij = distance(data,height,width,channels,x_col,x_row,y_col,y_row);
					float Fij = exp(-Dij / (2 * sigma * sigma));
					Ei += Fij;
				}
			}
			// normalize
			E[x_col + height * x_row] = Ei / ((y_col_max-y_col_min)*(y_row_max-y_row_min));
		}
	}

 	// find best neighbors
	for (int x_row = 0; x_row < width; ++x_row) {
		for (int x_col = 0; x_col < height; ++x_col) {

			// varibales for best neihbor
			float E0 = E[x_col + height * x_row];
			float d_best = INF;
			float y_col_best = x_col;
			float y_row_best = x_row; 

			// initialize boundaries from dist
			int y_col_min = MAX(x_col - Rd, 0);
			int y_col_max = MIN(x_col + Rd, height-1);
			int y_row_min = MAX(x_row - Rd, 0);
			int y_row_max = MIN(x_row + Rd, width-1);

			for (int y_row = y_row_min; y_row <= y_row_max; ++ y_row) {
				for (int y_col = y_col_min; y_col <= y_col_max; ++ y_col) {
					if (E[y_col + height * y_row] > E0) {
						float Dij = distance(data,height,width,channels, x_col,x_row, y_col,y_row);					 
						if (Dij <= dist*dist && Dij < d_best) {
							d_best = Dij;
							y_col_best = y_col;
							y_row_best = y_row;
						}
					}
				}
			}

			// map is the index of the best pair
			// gaps is the minimal distance, INF = root
			map[x_col + height * x_row] = y_col_best + height * y_row_best;
			if (map[x_col + height * x_row] != x_col + height * x_row) gaps[x_col + height * x_row] = sqrt(d_best);
			else gaps[x_col + height * x_row] = d_best;
		}
	}

	*time = seconds() - start;

}