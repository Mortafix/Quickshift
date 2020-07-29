// std import
#include <stdlib.h>
#include <string.h>
#include <fstream>
// cuda
#include <cuda_runtime_api.h>
#include <cuda.h>
// interface
#include "quickshift_cmn.h"
// read/write image
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
// cuda check and timer
#include "common.h"

qs_image image_segmentation(qs_image image, int * flatmap, int &roots){
	float * meancolor = (float *) calloc(image.height*image.width*image.channels, sizeof(float)) ;
	float * counts = (float *) calloc(image.height*image.width, sizeof(float)) ;
	// mean color
	for (int p = 0; p < image.height*image.width; p++){
		counts[flatmap[p]]++;
		for (int k = 0; k < image.channels; k++)
			meancolor[flatmap[p] + k*image.height*image.width] += image.data[p + k*image.height*image.width];
	}
	// count roots
	roots = 0;
	for (int p = 0; p < image.height*image.width; p++){
		if (flatmap[p] == p)
			roots++;
	}
	// check if different nonzero
	int nonzero = 0;
	for (int p = 0; p < image.height*image.width; p++){
		if (counts[p] > 0){
			nonzero++;
			for (int k = 0; k < image.channels; k++)
				meancolor[p + k*image.height*image.width] /= counts[p];
		}
	}
	if (roots != nonzero)
		printf("Nonzero: %d\n", nonzero);
	assert(roots == nonzero);
	// create output image
	qs_image imout = image;
	imout.data = (float *) calloc(image.height*image.width*image.channels, sizeof(float));
	for (int p = 0; p < image.height*image.width; p++)
		for (int k = 0; k < image.channels; k++)
			imout.data[p + k*image.height*image.width] = meancolor[flatmap[p] + k*image.height*image.width];
	// cleanup
	free(meancolor);
	free(counts);
	return imout;
}

int * map_to_flatmap(float * map, unsigned int size){
	// first level flatmap
	int *flatmap = (int *) malloc(size*sizeof(int)) ;
	for (unsigned int p = 0; p < size; p++)
		flatmap[p] = map[p];
	// flattern
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

void stbImage_to_QS(stbi_uc* pixels, int width, int height, int channels, qs_image & image){
	image.height = height;
	image.width = width;
	image.channels = channels;
	image.data = (float *) calloc(image.height*image.width*image.channels, sizeof(float));
	for(int k = 0; k < image.channels; k++)
		for(int col = 0; col < image.width; col++)
			for(int row = 0; row < image.height; row++){
				stbi_uc pixel = pixels[channels * (row * width + col) + k];
				image.data[row + col*image.height + k*image.height*image.width] = 32. * pixel / 255.; // Scale 0-32
			}
}

stbi_uc* QS_to_stbImage(qs_image image){
	stbi_uc * result = (stbi_uc *) calloc(image.height*image.width*image.channels, sizeof(stbi_uc));
	for(int k = 0; k < image.channels; k++)
		for(int col = 0; col < image.width; col++)
			for(int row = 0; row < image.height; row++){
				result[image.channels * (row * image.width + col) + k] = (stbi_uc) (image.data[row + col*image.height + k*image.height*image.width]/32*255); // scale 0-255
			}
	return result;
}

int main(int argc, char ** argv){

	// check options
	if (argc != 5 && argc != 6){
		printf("\nUSAGE: Quickshift <image> <mode>[cpu/gpu] <sigma> <dist> <texture_memory>[gpu:y/n]\n\n");
		exit(-1);
	}

	// grab options
	char *file = argv[1];
	const char *mode = argv[2];
	int sigma = atoi(argv[3]);
	int dist =  atoi(argv[4]);
	int texture = (argc == 6 && !strcmp(mode,"gpu") && !strcmp(argv[5],"y"));

	// read image
	qs_image image;
	int width, height, channels;
	stbi_uc* pixels = stbi_load(file, &width, &height, &channels, 0);
	printf("\n# Reading \'%s\' [%dx%d pxs, %d channel(s)]\n",file,width,height,channels);
	stbImage_to_QS(pixels,width,height,channels,image);

	// memory setup
	float *map, *E, *gaps;
	int * flatmap, roots;
	stbi_uc* out_pixels;
	qs_image imout;
	map = (float *) calloc(image.height*image.width, sizeof(float)) ;
	gaps = (float *) calloc(image.height*image.width, sizeof(float)) ;
	E = (float *) calloc(image.height*image.width, sizeof(float)) ;

	// # QUICKSHIFT #
	// printing
	const char *tex_mem = "", *mode_msg;
	if(!strcmp(mode,"cpu")) mode = "CPU"; else mode = "GPU";
	if(!strcmp(mode,"GPU")){ if(texture) tex_mem = "   Texture: Yes\n"; else tex_mem = "   Texture: No\n"; }
	printf("# Executing Quickshift in %s mode...\n   Sigma:   %d\n   Dist:    %d\n%s",mode,sigma,dist,tex_mem);
	// execution
	float time;
	if(!strcmp(mode,"CPU")) quickshift_cpu(image, sigma, dist, map, gaps, E, &time);
	else quickshift_gpu(image, sigma, dist, map, gaps, E, texture, &time);

	// consistency check
	for(int p = 0; p < image.height*image.width; p++)
		if(map[p] == p) assert(gaps[p] == INF);

	// output file name
	if(!strcmp(mode,"GPU") && texture) mode = "GPUt";
	char output[1024];
	sprintf(output, "%s", file);
	char * point = strrchr(output, '.');
	if(point) *point = '\0';
	sprintf(output, "%s-%s_%d-%d.jpg", output, mode, sigma, dist);

	// write output image
	flatmap = map_to_flatmap(map, image.height*image.width);
	imout = image_segmentation(image, flatmap, roots);
	out_pixels = QS_to_stbImage(imout);
	stbi_write_jpg(output, width, height, channels, out_pixels, 100);
	printf("# Complete\n    Elapsed time:  %f sec\n    Roots:         %d\n", time, roots);

	// cleanup
	printf("\n");
	free(flatmap);
	free(imout.data);
	free(image.data);
	free(map);
	free(E);
	free(gaps);
	stbi_image_free(pixels);
}
