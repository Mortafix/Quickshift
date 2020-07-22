#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include "quickshift_cmn.h"

texture<float, 3, cudaReadModeElementType> texI;
texture<float, 2, cudaReadModeElementType> texE;

#define USE_TEX_E 0
#define USE_TEX_I 0

#if USE_TEX_I
	#define TEXI(x,y,c) tex3D(texI, x + 0.5f, y + 0.5f, c + 0.5f)
#else
	#define TEXI(x,y,c) data [ (x) + height*(y) + width*height*k ]
#endif

#if USE_TEX_E
	#define TEXE(x,y) tex2D(texE, x + 0.5f, y + 0.5f)
#else
	#define TEXE(x,y) E [ (x) + height* (y)]
#endif

#define distance(data,height,width,channels,v,j1,j2,dist)			\
{																	\
	dist = 0;														\
	int d1 = j1 - i1;												\
	int d2 = j2 - i2;												\
	int k;															\
	dist += d1*d1 + d2*d2;										 	\
	for (k = 0; k < channels; ++k) {								\
		float d =	v[k] - TEXI(j1,j2,k);							\
		dist += d*d;												\
	}																\
}

extern "C"
int iDivUp(int num, int denom){
	return (num % denom != 0) ? (num / denom + 1) : (num / denom);
}


extern "C"
__global__ void find_neighbors_gpu(const float * data, int height, int width, int channels, float * E, float dist, int Rd, float * map, float * gaps){	 
	int i1 = blockIdx.y * blockDim.y + threadIdx.y;
	int i2 = blockIdx.x * blockDim.x + threadIdx.x;
	if (i1 >= height || i2 >= width) return; // out of bounds

	int j1,j2;
			
	// varibales for best neihbor	
	float E0 = TEXE(i1, i2);
	float d_best = INF;
	float j1_best = i1	;
	float j2_best = i2	; 
	
	// initialize boundaries from dist
	int j1min = MAX(i1 - Rd, 0	 );
	int j1max = MIN(i1 + Rd, height-1);
	int j2min = MAX(i2 - Rd, 0	 );
	int j2max = MIN(i2 + Rd, width-1);			
 
	// cache the center value in registers
	float v[3];
	for (int k = 0; k < channels; ++k) {
		v[k] =	TEXI(i1,i2,k);
	}

	for (j2 = j2min; j2 <= j2max; ++ j2) {
		for (j1 = j1min; j1 <= j1max; ++ j1) {						
			if (TEXE(j1,j2) > E0) {
				float Dij;
				distance(data,height,width,channels, v, j1,j2,Dij);
				if (Dij <= dist*dist && Dij < d_best) {
					d_best = Dij;
					j1_best = j1;
					j2_best = j2;
				}
			}
		}
	}
	
	// map is the index of the best pair
	// gaps is the minimal distance, INF = root
	map [i1 + height * i2] = j1_best + height * j2_best; /* + 1; */
	if (map[i1 + height * i2] != i1 + height * i2) gaps[i1 + height * i2] = sqrt(d_best);
	else gaps[i1 + height * i2] = d_best;
}

extern "C"
__global__ void compute_E_gpu(const float * data, int height, int width, int channels, int R, float sigma, float * E, float * n, float * M){
	
	// thread index
	int i1 = blockIdx.y * blockDim.y + threadIdx.y;
	int i2 = blockIdx.x * blockDim.x + threadIdx.x;
	if (i1 >= height || i2 >= width) return; // out of bounds
	int j1,j2;
	
	// initialize boundaries from sigma
	int j1min = MAX(i1 - R, 0	 );
	int j1max = MIN(i1 + R, height-1);
	int j2min = MAX(i2 - R, 0	 );
	int j2max = MIN(i2 + R, width-1);			
	float Ei = 0;

	// cache the center value in registers
	float v[3];
	for (int k = 0; k < channels; ++k) {
		v[k] = TEXI(i1,i2,k);
	}

	// for each pixel in the area (sigma) compute the distance between it and the source pixel
	for (j2 = j2min; j2 <= j2max; ++ j2) {
		for (j1 = j1min; j1 <= j1max; ++ j1) {
			float Dij;
			distance(data, height, width, channels,v ,j1, j2, Dij);
			float Fij = - exp(- Dij / (2*sigma*sigma));
			Ei += -Fij;
		}
	}
	// normalize
	E [i1 + height * i2] = Ei / ((j1max-j1min)*(j2max-j2min));
}


extern "C" 
void quickshift_gpu(qs_image image, float sigma, float dist, float * map, float * gaps, float * E){
/*#if USE_TEX_I
	printf("quickshiftGPU: using texture for I\n");
	cudaArray * cu_array_I;

	// Allocate array
	cudaChannelFormatDesc descriptionI = cudaCreateChannelDesc<float>();

	cudaExtent const ext = {image.height, image.width, image.channels};
	cudaMalloc3DArray(&cu_array_I, &descriptionI, ext);

	cudaMemcpy3DParms copyParams = {0};
	copyParams.extent = make_cudaExtent(image.height, image.width, image.channels);
	copyParams.kind = cudaMemcpyHostToDevice;
	copyParams.dstArray = cu_array_I;
	// The pitched pointer is really tricky to get right. We give the
	// pitch of a row, then the number of elements in a row, then the
	// height, and we omit the 3rd dimension.
	copyParams.srcPtr = make_cudaPitchedPtr(
	(void*)&image.data[0], ext.width*sizeof(float), ext.width, ext.height);
	cudaMemcpy3D(&copyParams);

	cudaBindTextureToArray(texI, cu_array_I, descriptionI);

	texI.normalized = false;
	texI.filterMode = cudaFilterModePoint;
#endif*/


	float *map_d, *E_d, *gaps_d, *data;
	
	int height = image.height;
	int width = image.width;
	int channels = image.channels;

	unsigned int size = image.height*image.width * sizeof(float);
	cudaMalloc( (void**) &data, size*image.channels);
	cudaMalloc( (void**) &map_d, size);
	cudaMalloc( (void**) &gaps_d, size);
	cudaMalloc( (void**) &E_d, size);

	cudaMemcpy( data, image.data, size*image.channels, cudaMemcpyHostToDevice);
	cudaMemset( E_d, 0, size);

	int R = (int) ceil (3 * sigma);
	int Rd = (int) ceil (dist);
	
	dim3 dimBlock(32,4,1);
	dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);
	compute_E_gpu <<<dimGrid,dimBlock>>> (data, height, width, channels, R, sigma, E_d, 0, 0);

	cudaThreadSynchronize();

	cudaMemcpy(E, E_d, size, cudaMemcpyDeviceToHost);


	/* Texture map E */
/*#if USE_TEX_E
	printf("quickshiftGPU: using texture for E\n");
	cudaChannelFormatDesc descriptionE = cudaCreateChannelDesc<float>();

	cudaArray * cu_array_E;
	cudaMallocArray(&cu_array_E, &descriptionE, image.height, image.width);

	cudaMemcpyToArray(cu_array_E, 0, 0, E, sizeof(float)*image.height*image.width,
					cudaMemcpyHostToDevice);

	texE.normalized = false;
	texE.filterMode = cudaFilterModePoint;

	cudaBindTextureToArray(texE, cu_array_E,
				descriptionE);

	cudaThreadSynchronize();
#endif*/

	/* -----------------------------------------------------------------
	 *																							 Find best neighbors
	 * -------------------------------------------------------------- */
	
	find_neighbors_gpu <<<dimGrid,dimBlock>>> (data, height ,width, channels, E_d, dist, Rd, map_d, gaps_d);

	cudaThreadSynchronize();
	
	cudaMemcpy(map, map_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(gaps, gaps_d, size, cudaMemcpyDeviceToHost);


	cudaFree(data);
	cudaFree(map_d);
	cudaFree(gaps_d);
	cudaFree(E_d);
	//cudaUnbindTexture(texI);
	//cudaFreeArray(cu_array_I);
	//cudaUnbindTexture(texE);
	//cudaFreeArray(cu_array_E);

}
