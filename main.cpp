#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "quickshift_common.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

image_t imseg(image_t im, int * flatmap){
	// mean Color
	float * meancolor = (float *) calloc(im.N1*im.N2*im.K, sizeof(float)) ;
	float * counts = (float *) calloc(im.N1*im.N2, sizeof(float)) ;

	for (int p = 0; p < im.N1*im.N2; p++){
		counts[flatmap[p]]++;
		for (int k = 0; k < im.K; k++)
			meancolor[flatmap[p] + k*im.N1*im.N2] += im.I[p + k*im.N1*im.N2];
	}

	int roots = 0;
	for (int p = 0; p < im.N1*im.N2; p++){
		if (flatmap[p] == p)
			roots++;
	}
	printf("Roots: %d\n", roots);

	int nonzero = 0;
	for (int p = 0; p < im.N1*im.N2; p++){
		if (counts[p] > 0){
			nonzero++;
			for (int k = 0; k < im.K; k++)
				meancolor[p + k*im.N1*im.N2] /= counts[p];
		}
	}
	if (roots != nonzero)
		printf("Nonzero: %d\n", nonzero);
	assert(roots == nonzero);

	// create output image
	image_t imout = im;
	imout.I = (float *) calloc(im.N1*im.N2*im.K, sizeof(float));
	for (int p = 0; p < im.N1*im.N2; p++)
		for (int k = 0; k < im.K; k++)
			imout.I[p + k*im.N1*im.N2] = meancolor[flatmap[p] + k*im.N1*im.N2];

	free(meancolor);
	free(counts);

	return imout;
}

int * map_to_flatmap(float * map, unsigned int size){
	// flatmap
	int *flatmap = (int *) malloc(size*sizeof(int)) ;
	for (unsigned int p = 0; p < size; p++)
		flatmap[p] = map[p];

	bool changed = true;
	while (changed){
		changed = false;
		for (unsigned int p = 0; p < size; p++){
			changed = changed || (flatmap[p] != flatmap[flatmap[p]]);
			flatmap[p] = flatmap[flatmap[p]];
		}
	}

	// consistency check
	for (unsigned int p = 0; p < size; p++)
		assert(flatmap[p] == flatmap[flatmap[p]]);

	return flatmap;
}

void stbImage_to_QS(stbi_uc* pixels, int width, int height, int channels, image_t & im){
	im.N1 = height;
	im.N2 = width;
	im.K = channels;
	im.I = (float *) calloc(im.N1*im.N2*im.K, sizeof(float));
	for(int k = 0; k < im.K; k++)
		for(int col = 0; col < im.N2; col++)
			for(int row = 0; row < im.N1; row++){
				stbi_uc pixel = pixels[channels * (row * width + col) + k];
				im.I[row + col*im.N1 + k*im.N1*im.N2] = 32. * pixel / 255.; // Scale 0-32
			}
}

stbi_uc* QS_to_stbImage(image_t im){
	stbi_uc * result = (stbi_uc *) calloc(im.N1*im.N2*im.K, sizeof(stbi_uc));
	for(int k = 0; k < im.K; k++)
		for(int col = 0; col < im.N2; col++)
			for(int row = 0; row < im.N1; row++){
				result[im.K * (row * im.N2 + col) + k] = (stbi_uc) (im.I[row + col*im.N1 + k*im.N1*im.N2]/32*255); // scale 0-255
			}
	return result;
}

int main(int argc, char ** argv){

	// check options
	if (argc != 5){
		printf("USAGE: Quickshift <image> <mode>[cpu/gpu] <sigma> <tau>\n\n");
		exit(-1);
	}

	// grab options
	char *file = argv[1];
	char *mode = argv[2];
	float sigma = ::atof(argv[3]);
	float tau = ::atof(argv[4]);

	// read image
	image_t im;
	int width, height, channels;
	stbi_uc* pixels = stbi_load(file, &width, &height, &channels, 0);
	printf("\n# Reading \'%s\' [%dx%d pxs] and %d channel(s)\n",file,width,height,channels);
	stbImage_to_QS(pixels,width,height,channels,im);

	// memory setup
	float *map, *E, *gaps;
	int * flatmap;
	stbi_uc* out_pixels;
	image_t imout;
	map = (float *) calloc(im.N1*im.N2, sizeof(float)) ;
	gaps = (float *) calloc(im.N1*im.N2, sizeof(float)) ;
	E = (float *) calloc(im.N1*im.N2, sizeof(float)) ;

	// QUICKSHIFT
	if(!strcmp(mode,"cpu")){
		quickshift_cpu(im, sigma, tau, map, gaps, E);
	} else if(!strcmp(mode,"gpu")){
		quickshift_gpu(im, sigma, tau, map, gaps, E);
	} else { printf("Mode must be cpu or gpu.\n"); exit(-1); }

	printf("# Executing Quickshift in %s mode...\nSigma: %.1f\nTau: %.1f\n",mode,sigma,tau);

	// consistency check
	for(int p = 0; p < im.N1*im.N2; p++)
		if(map[p] == p) assert(gaps[p] == INF);

	// output file name
	char output[1024];
	sprintf(output, "%s", file);
	char * point = strrchr(output, '.');
	if(point) *point = '\0';
	sprintf(output, "%s-%s_%.0f-%.0f.jpg", output, mode, sigma, tau);

	// write output image
	flatmap = map_to_flatmap(map, im.N1*im.N2);
	imout = imseg(im, flatmap);
	out_pixels = QS_to_stbImage(imout);
	stbi_write_jpg(output, width, height, channels, out_pixels, 100);
	
	// cleanup
	printf("\n");
	free(flatmap);
	free(imout.I);
	free(im.I);
	free(map);
	free(E);
	free(gaps);
	stbi_image_free(pixels);
}
