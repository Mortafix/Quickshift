#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "quickshift_cmn.h"

/** -----------------------------------------------------------------
 ** @internal
 ** @brief Computes the accumulated channel L2 distance between i,j + the distance between i,j
 **
 ** @param data    input image buffer
 ** @param height   size of the first dimension of the image
 ** @param width   size of the second dimension of the image
 ** @param channels    number of channels
 ** @param i1   first dimension index of the first pixel to compare
 ** @param i2   second dimension of the first pixel
 ** @param j1   index of the second pixel to compare
 ** @param j2   second dimension of the second pixel
 **
 ** Takes the L2 distance between the values in data at pixel i and j, accumulating along channels channels
 ** and adding in the distance between i,j in the image.
 ** 
 ** @return the distance as described above
 **/

inline
float
distance(float const * data, 
         int height, int width, int channels,
         int i1, int i2,
         int j1, int j2) 
{
  float dist = 0 ;
  int d1 = j1 - i1 ;
  int d2 = j2 - i2 ;
  int k ;
  dist += d1*d1 + d2*d2 ;
  // For k = 0...channels-1, d+= L2 distance between data(i1,i2,k) and data(j1,j2,k)
  for (k = 0 ; k < channels ; ++k) {
    float d = 
      data [i1 + height * i2 + (height*width) * k] - 
      data [j1 + height * j2 + (height*width) * k] ;
    dist += d*d ;
  }
  return dist ;
}

/** -----------------------------------------------------------------
 ** @internal
 ** @brief Computes the accumulated channel inner product between i,j + the
 **        distance between i,j
 ** 
 ** @param data    input image buffer
 ** @param height   size of the first dimension of the image
 ** @param width   size of the second dimension of the image
 ** @param channels    number of channels
 ** @param i1   first dimension index of the first pixel to compare
 ** @param i2   second dimension of the first pixel
 ** @param j1   index of the second pixel to compare
 ** @param j2   second dimension of the second pixel
 **
 ** Takes the channel-wise inner product between the values in data at pixel i and
 ** j, accumulating along channels channels and adding in the inner product between i,j in
 ** the image.
 ** 
 ** @return the inner product as described above
 **/

inline
float
inner(float const * data, 
      int height, int width, int channels,
      int i1, int i2,
      int j1, int j2) 
{
  float ker = 0 ;
  int k ;
  ker += i1*j1 + i2*j2 ;
  for (k = 0 ; k < channels ; ++k) {
    ker += 
      data [i1 + height * i2 + (height*width) * k] *
      data [j1 + height * j2 + (height*width) * k] ;
  }
  return ker ;
}


void quickshift_cpu(qs_image image, float sigma, float dist, float * map, float * gaps, float * E)
{
  int verb = 1 ;

  float *M = 0, *n = 0;
  float dist2;
  
  int channels, d;
  int height,width, i1,i2, j1,j2, R, tR;

  int medoid = 0 ;

  float const * data = image.data;
  height = image.height;
  width = image.width;
  channels = image.channels;

  d = 2 + channels ; /* Total dimensions include spatial component (x,y) */
  
  dist2  = dist*dist;

  
  if (medoid) { /* n and M are only used in mediod shift */
    M = (float *) calloc(height*width*d, sizeof(float)) ;
    n = (float *) calloc(height*width,   sizeof(float)) ;
  }

  R = (int) ceil (3 * sigma) ;
  tR = (int) ceil (dist) ;
  
  if (verb) {
    printf("quickshift: [height,width,channels]: [%d,%d,%d]\n", height,width,channels) ;
    printf("quickshift: type: %s\n", medoid ? "medoid" : "quick");
    printf("quickshift: sigma:   %g\n", sigma) ;
    /* R is ceil(3 * sigma) and determines the window size to accumulate
     * similarity */
    printf("quickshift: R:       %d\n", R) ; 
    printf("quickshift: dist:     %g\n", dist) ;
    printf("quickshift: tR:      %d\n", tR) ;
  }

  /* -----------------------------------------------------------------
   *                                                                 n 
   * -------------------------------------------------------------- */

  /* If we are doing medoid shift, initialize n to the inner product of the
   * image with itself
   */
  if (n) { 
    for (i2 = 0 ; i2 < width ; ++ i2) {
      for (i1 = 0 ; i1 < height ; ++ i1) {        
        n [i1 + height * i2] = inner(data,height,width,channels,
                                 i1,i2,
                                 i1,i2) ;
      }
    }
  }
  
  unsigned int Etimer;
  /*cutilCheckError( cutCreateTimer(&Etimer) );
  cutilCheckError( cutResetTimer(Etimer) );
  cutilCheckError( cutStartTimer(Etimer) );*/

  /* -----------------------------------------------------------------
   *                                                 E = - [oN'*F]', M
   * -------------------------------------------------------------- */
  
  /* 
     D_ij = d(x_i,x_j)
     E_ij = exp(- .5 * D_ij / sigma^2) ;
     F_ij = - E_ij             
     E_i  = sum_j E_ij
     M_di = sum_j X_j F_ij

     E is the parzen window estimate of the density
     0 = dissimilar to everything, windowsize = identical
  */
  
  for (i2 = 0 ; i2 < width ; ++ i2) {
    for (i1 = 0 ; i1 < height ; ++ i1) {
      
      float Ei = 0;
      int j1min = MAX(i1 - R, 0   ) ;
      int j1max = MIN(i1 + R, height-1) ;
      int j2min = MAX(i2 - R, 0   ) ;
      int j2max = MIN(i2 + R, width-1) ;      
      
      /* For each pixel in the window compute the distance between it and the
       * source pixel */
      for (j2 = j2min ; j2 <= j2max ; ++ j2) {
        for (j1 = j1min ; j1 <= j1max ; ++ j1) {
          float Dij = distance(data,height,width,channels, i1,i2, j1,j2) ;          
          /* Make distance a similarity */ 
          float Fij = exp(- Dij / (2*sigma*sigma)) ;

          /* E is E_i above */
          Ei += Fij;
          
          if (M) {
            /* Accumulate votes for the median */
            int k ;
            M [i1 + height*i2 + (height*width) * 0] += j1 * Fij ;
            M [i1 + height*i2 + (height*width) * 1] += j2 * Fij ;
            for (k = 0 ; k < channels ; ++k) {
              M [i1 + height*i2 + (height*width) * (k+2)] += 
                data [j1 + height*j2 + (height*width) * k] * Fij ;
            }
          } 
          
        } /* j1 */ 
      } /* j2 */
      /* Normalize */
      E [i1 + height * i2] = Ei / ((j1max-j1min)*(j2max-j2min));
      
      /*E [i1 + height * i2] = Ei ; */

    }  /* i1 */
  } /* i2 */
  
  /*cutilCheckError( cutStopTimer(Etimer) );
  float ETime = cutGetTimerValue(Etimer);
  printf("ComputeE: %fms\n", ETime);*/

  /*unsigned int Ntimer;
  cutilCheckError( cutCreateTimer(&Ntimer) );
  cutilCheckError( cutResetTimer(Ntimer) );
  cutilCheckError( cutStartTimer(Ntimer) );*/
 
  /* -----------------------------------------------------------------
   *                                               Find best neighbors
   * -------------------------------------------------------------- */
  
  if (medoid) {
    
    /* 
       Qij = - nj Ei - 2 sum_k Gjk Mik
       n is data.^2
    */
    
    /* medoid shift */
    for (i2 = 0 ; i2 < width ; ++i2) {
      for (i1 = 0 ; i1 < height ; ++i1) {
        
        float sc_best = 0  ;
        /* j1/j2 best are the best indicies for each i */
        float j1_best = i1 ;
        float j2_best = i2 ; 
        
        int j1min = MAX(i1 - R, 0   ) ;
        int j1max = MIN(i1 + R, height-1) ;
        int j2min = MAX(i2 - R, 0   ) ;
        int j2max = MIN(i2 + R, width-1) ;      
        
        for (j2 = j2min ; j2 <= j2max ; ++ j2) {
          for (j1 = j1min ; j1 <= j1max ; ++ j1) {            
            
            float Qij = - n [j1 + j2 * height] * E [i1 + i2 * height] ;
            int k ;

            Qij -= 2 * j1 * M [i1 + i2 * height + (height*width) * 0] ;
            Qij -= 2 * j2 * M [i1 + i2 * height + (height*width) * 1] ;
            for (k = 0 ; k < channels ; ++k) {
              Qij -= 2 * 
                data [j1 + j2 * height + (height*width) * k] *
                M [i1 + i2 * height + (height*width) * (k + 2)] ;
            }
            
            if (Qij > sc_best) {
              sc_best = Qij ;
              j1_best = j1 ;
              j2_best = j2 ;
            }
          }
        }

        /* map_i is the linear index of j which is the best pair (in matlab
         * notation
         * gaps_i is the score of the best match
         */
        map [i1 + height * i2] = j1_best + height * j2_best ; /*+ 1 ; */
        gaps[i1 + height * i2] = sc_best ;
      }
    }  

  } else {
    
    /* Quickshift assigns each i to the closest j which has an increase in the
     * density (E). If there is no j s.t. Ej > Ei, then gaps_i == inf (a root
     * node in one of the trees of merges).
     */
    for (i2 = 0 ; i2 < width ; ++i2) {
      for (i1 = 0 ; i1 < height ; ++i1) {
        
        float E0 = E [i1 + height * i2] ;
        float d_best = INF ;
        float j1_best = i1   ;
        float j2_best = i2   ; 
        
        int j1min = MAX(i1 - tR, 0   ) ;
        int j1max = MIN(i1 + tR, height-1) ;
        int j2min = MAX(i2 - tR, 0   ) ;
        int j2max = MIN(i2 + tR, width-1) ;      
        
        for (j2 = j2min ; j2 <= j2max ; ++ j2) {
          for (j1 = j1min ; j1 <= j1max ; ++ j1) {            
            if (E [j1 + height * j2] > E0) {
              float Dij = distance(data,height,width,channels, i1,i2, j1,j2) ;           
              if (Dij <= dist2 && Dij < d_best) {
                d_best = Dij ;
                j1_best = j1 ;
                j2_best = j2 ;
              }
            }
          }
        }
        
        /* map is the index of the best pair */
        /* gaps_i is the minimal distance, inf implies no Ej > Ei within
         * distance dist from the point */
        map [i1 + height * i2] = j1_best + height * j2_best ; /* + 1 ; */
        if (map[i1 + height * i2] != i1 + height * i2)
          gaps[i1 + height * i2] = sqrt(d_best) ;
        else
          gaps[i1 + height * i2] = d_best; /* inf */
      }
    }  
  }
  
  if (M) free(M) ;
  if (n) free(n) ;
  
  /*cutilCheckError( cutStopTimer(Ntimer) );
  float NTime = cutGetTimerValue(Ntimer);
  printf("ComputeN: %fms\n", NTime);*/

}


