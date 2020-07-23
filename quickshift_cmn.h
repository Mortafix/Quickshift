#ifndef __QUICKSHIFT_CMN_H__
#define __QUICKSHIFT_CMN_H__
#include <float.h>

typedef unsigned int vl_uint32 ;
typedef unsigned char vl_uint8 ; 
typedef unsigned short vl_uint16 ;

#define INF FLT_MAX
#define MIN(a,b) ( ((a) <  (b) ) ? (a) : (b) )
#define MAX(a,b) ( ((a) >  (b) ) ? (a) : (b) )
#define ABS(a)   ( ((a) >= 0   ) ? (a) :-(a) )

typedef struct _qs_image {
  float * data;
  int height, width, channels;
} qs_image;

void quickshift_cpu(qs_image image, float sigma, float dist, float * map, float * gaps, float * E);
void quickshift_gpu(qs_image image, float sigma, float dist, float * map, float * gaps, float * E, int texture);

#endif
