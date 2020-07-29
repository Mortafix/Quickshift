#include <stdio.h>
#include <stdlib.h>
#include "quickshift_cmn.h"
#include "common.h"

texture<float, 3, cudaReadModeElementType> texture_pixels;
texture<float, 2, cudaReadModeElementType> texture_density;

__device__ float get_pixel(int with_texture, int x, int y, int ch, int height, int width, const float * data){
	if(with_texture) return tex3D(texture_pixels, x, y, ch);
	else return data[x + height*y + width*height*ch];
}

__device__ float get_density(int with_texture, int x, int y, int height, float * E){
	if(with_texture) return tex2D(texture_density, x, y);
	else return E[x + height*y];
}

__device__ float distance(const float * data, int height, int width, int channels, float * v, int x_col, int x_row, int y_col, int y_row, int with_texture){
	int d1 = y_col - x_col;
	int d2 = y_row - x_row;
	int k;
	float dist = d1*d1 + d2*d2;										 
	for (k = 0; k < channels; ++k) {
		float d = v[k] - get_pixel(with_texture,y_col,y_row,k,height,width,data);
		dist += d*d;
	}
	return dist;
}

int divide_grid(int num, int den){
	return (num % den != 0) ? (num / den + 1) : (num / den);
}


__global__ void find_neighbors(const float * data, int height, int width, int channels, float * E, float dist, int Rd, float * map, float * gaps, int with_texture){	 
	
	// thread index
	int x_col = blockIdx.y * blockDim.y + threadIdx.y;
	int x_row = blockIdx.x * blockDim.x + threadIdx.x;
	if (x_col >= height || x_row >= width) return; // out of bounds

	// varibales for best neighbor
	int y_col,y_row;
	float E0 = get_density(with_texture,x_col,x_row,height,E);
	float d_best = INF;
	float y_col_best = x_col;
	float y_row_best = x_row; 

	// initialize boundaries from dist
	int y_col_min = MAX(x_col - Rd, 0);
	int y_col_max = MIN(x_col + Rd, height-1);
	int y_row_min = MAX(x_row - Rd, 0);
	int y_row_max = MIN(x_row + Rd, width-1);
 
	// cache the center value
	float v[3];
	for (int k = 0; k < channels; ++k)
		v[k] = get_pixel(with_texture,x_col,x_row,k,height,width,data);

	for (y_row = y_row_min; y_row <= y_row_max; ++ y_row) {
		for (y_col = y_col_min; y_col <= y_col_max; ++ y_col) {
			if (get_density(with_texture,y_col,y_row,height,E) > E0) {
				float Dij = distance(data,height,width,channels,v,x_col,x_row,y_col,y_row,with_texture);
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
	map [x_col + height * x_row] = y_col_best + height * y_row_best;
	if (map[x_col + height * x_row] != x_col + height * x_row) gaps[x_col + height * x_row] = sqrt(d_best);
	else gaps[x_col + height * x_row] = d_best;
}

__global__ void compute_density(const float * data, int height, int width, int channels, int R, float sigma, float * E, int with_texture){

	// thread index
	int x_col = blockIdx.y * blockDim.y + threadIdx.y;
	int x_row = blockIdx.x * blockDim.x + threadIdx.x;
	if (x_col >= height || x_row >= width) return; // out of bounds

	// initialize boundaries from sigma
	int y_col,y_row;
	int y_col_min = MAX(x_col - R, 0);
	int y_col_max = MIN(x_col + R, height-1);
	int y_row_min = MAX(x_row - R, 0);
	int y_row_max = MIN(x_row + R, width-1);
	float Ei = 0;

	// cache the center value in registers
	float v[3];
	for (int k = 0; k < channels; ++k)
		v[k] = get_pixel(with_texture,x_col,x_row,k,height,width,data);

	// for each pixel in the area (sigma) compute the distance between it and the source pixel
	for (y_row = y_row_min; y_row <= y_row_max; ++ y_row) {
		for (y_col = y_col_min; y_col <= y_col_max; ++ y_col) {
			float Dij = distance(data,height,width,channels,v,x_col,x_row,y_col,y_row,with_texture);
			float Fij = exp(-Dij / (2*sigma*sigma));
			Ei += Fij;
		}
	}
	// normalize
	E[x_col + height * x_row] = Ei / ((y_col_max-y_col_min)*(y_row_max-y_row_min));
}


void quickshift_gpu(qs_image image, float sigma, float dist, float * map, float * gaps, float * E, int with_texture, float * time){

	CHECK( cudaSetDevice(2) );

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaArray * cuda_array_pixels;
	cudaArray * cuda_array_density;

	// texture for the image
	if(with_texture){

		cudaChannelFormatDesc descr_pixels = cudaCreateChannelDesc<float>();

		texture_pixels.normalized = false;
		texture_pixels.filterMode = cudaFilterModePoint;

		cudaExtent const ext = {image.height, image.width, image.channels};
		CHECK( cudaMalloc3DArray(&cuda_array_pixels, &descr_pixels, ext) );

		cudaMemcpy3DParms copyParams = {0};
		copyParams.extent = make_cudaExtent(image.height, image.width, image.channels);
		copyParams.kind = cudaMemcpyHostToDevice;
		copyParams.dstArray = cuda_array_pixels;
		copyParams.srcPtr = make_cudaPitchedPtr((void*)&image.data[0], ext.width*sizeof(float), ext.width, ext.height);
		CHECK( cudaMemcpy3D(&copyParams) );

		CHECK( cudaBindTextureToArray(texture_pixels, cuda_array_pixels, descr_pixels) );

	}

	// variables
	float *map_cuda, *E_cuda, *gaps_cuda, *data;
	int height = image.height;
	int width = image.width;
	int channels = image.channels;
	int R = (int) ceil (3 * sigma);
	int Rd = (int) ceil (dist);

	// allocate memory on device
	unsigned int size = image.height*image.width * sizeof(float);
	CHECK( cudaMalloc((void**) &data, size*image.channels) );
	CHECK( cudaMalloc((void**) &map_cuda, size) );
	CHECK( cudaMalloc((void**) &gaps_cuda, size) );
	CHECK( cudaMalloc((void**) &E_cuda, size) );

	CHECK( cudaMemcpy(data, image.data, size*image.channels, cudaMemcpyHostToDevice) );
	CHECK( cudaMemset(E_cuda, 0, size) );

	// compute density (and copy result to host)
	dim3 dimBlock(16,16,1);
	dim3 dimGrid(divide_grid(width, dimBlock.x), divide_grid(height, dimBlock.y), 1);
	CHECK( cudaEventRecord(start) );
	compute_density <<<dimGrid,dimBlock>>> (data, height, width, channels, R, sigma, E_cuda,with_texture);
	CHECK( cudaThreadSynchronize() );
	CHECK( cudaMemcpy(E, E_cuda, size, cudaMemcpyDeviceToHost) );

	// texture for density
	if(with_texture){

		cudaChannelFormatDesc descr_density = cudaCreateChannelDesc<float>();

		texture_density.normalized = false;
		texture_density.filterMode = cudaFilterModePoint;

		CHECK( cudaMallocArray(&cuda_array_density, &descr_density, image.height, image.width) );
		CHECK( cudaMemcpyToArray(cuda_array_density, 0, 0, E, sizeof(float)*image.height*image.width, cudaMemcpyHostToDevice) );

		CHECK( cudaBindTextureToArray(texture_density, cuda_array_density, descr_density) );

		CHECK( cudaThreadSynchronize() );
	}

	// find neighbors (and copy result to host)
	find_neighbors <<<dimGrid,dimBlock>>> (data, height ,width, channels, E_cuda, dist, Rd, map_cuda, gaps_cuda, with_texture);
	CHECK( cudaThreadSynchronize() );
	CHECK( cudaMemcpy(map, map_cuda, size, cudaMemcpyDeviceToHost) );
	CHECK( cudaMemcpy(gaps, gaps_cuda, size, cudaMemcpyDeviceToHost) );

	// time elapsed
	CHECK( cudaEventRecord(stop) );
	CHECK( cudaEventSynchronize(stop) );
	cudaEventElapsedTime(time, start, stop);
	*time /= 1000.0;

	// cleanup
	CHECK( cudaFree(data) );
	CHECK( cudaFree(map_cuda) );
	CHECK( cudaFree(gaps_cuda) );
	CHECK( cudaFree(E_cuda) );
	CHECK( cudaUnbindTexture(texture_pixels) );
	CHECK( cudaUnbindTexture(texture_density) );
	if(with_texture){
		CHECK( cudaFreeArray(cuda_array_pixels) );
		CHECK( cudaFreeArray(cuda_array_density) );
	}
}
