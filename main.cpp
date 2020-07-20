#include <stdlib.h>
#include <string.h>
#include "Image.h"
#include "Exception.h"
#include <fstream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "quickshift_common.h"

void write_image(image_t im, const char * filename){
	// copy from matlab style
	Image IMGOUT(im.K > 1 ? Image::RGB : Image::L, im.N2, im.N1);
	for(int k = 0; k < im.K; k++)
		for(int col = 0; col < im.N2; col++)
			for(int row = 0; row < im.N1; row++){
				// row transpose
				unsigned char * pt = IMGOUT.getPixelPt(col, im.N1-1-row);
				// scale 0-255
				pt[k] = (unsigned char) (im.I[row + col*im.N1 + k*im.N1*im.N2]/32*255);
			}
	// write image
	std::ofstream ofs(filename, std::ios::binary);
	if (!ofs) {
			throw Exception("Could not open the file");
	}
	ofs<<IMGOUT;
}

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

void image_to_matlab(Image & IMG, image_t & im){
	// convert image to MATLAB style representation
	im.N1 = IMG.getHeight();
	im.N2 = IMG.getWidth();
	im.K	= IMG.getPixelSize();
	im.I = (float *) calloc(im.N1*im.N2*im.K, sizeof(float));
	for(int k = 0; k < im.K; k++)
		for(int col = 0; col < im.N2; col++)
			for(int row = 0; row < im.N1; row++){
				unsigned char * pt = IMG.getPixelPt(col, im.N1-1-row);
				im.I[row + col*im.N1 + k*im.N1*im.N2] = 32. * pt[k] / 255.; // Scale 0-32
			}
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
	Image IMG;
	std::ifstream ifs(file, std::ios::binary);
	if (!ifs) {
		throw Exception("Could not open the file");
	}
	ifs>>IMG;
	image_t im;
	image_to_matlab(IMG, im);

	// memory setup
	float *map, *E, *gaps;
	int * flatmap;
	image_t imout;
	map = (float *) calloc(im.N1*im.N2, sizeof(float)) ;
	gaps = (float *) calloc(im.N1*im.N2, sizeof(float)) ;
	E = (float *) calloc(im.N1*im.N2, sizeof(float)) ;

	// QUICKSHIFT
	if(!strcmp(mode,"cpu")){
		printf("\n# Executing Quickshift in CPU mode...\nInput image: %s\nSigma: %.1f\nTau: %.1f\n",file,sigma,tau);
		quickshift(im, sigma, tau, map, gaps, E);
	} else if(!strcmp(mode,"gpu")){
		printf("\n# Executing Quickshift in GPU mode...\nInput image: %s\nSigma: %.1f\nTau: %.1f\n",file,sigma,tau);
		quickshift_gpu(im, sigma, tau, map, gaps, E);
	} else printf("Mode must be cpu or gpu.\n");

	// consistency check
	for(int p = 0; p < im.N1*im.N2; p++)
		if(map[p] == p) assert(gaps[p] == INF);

	// output image
	flatmap = map_to_flatmap(map, im.N1*im.N2);
	imout = imseg(im, flatmap);
	
	// writing output file name
	char output[1024];
	sprintf(output, "%s", file);
	char * point = strrchr(output, '.');
	if(point) *point = '\0';
	sprintf(output, "%s-%s_%.0f-%.0f.pnm", output, mode, sigma, tau);

	// writing image
	write_image(imout, output);

	// cleanup
	free(flatmap);
	free(imout.I);
	free(im.I);
	free(map);
	free(E);
	free(gaps);
}
