#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "matrix_lib.h"
#define THREADS_PER_BLOCK 256

// Kernel function to add the elements of two arrays
__global__ 
void mult_scalar(int n , float scalar, float *d_x){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x *blockDim.x;

    //printf("GLOBAL MULT SCALAR: %p ", d_x);

    for(int i = index; i < n; i += stride){
       d_x[i] = d_x[i] * scalar;

    }

}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix){
	long unsigned int h;
	long unsigned int w;

	h = matrix->height;
	w = matrix->width;

	if(matrix == NULL) return 0;

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (h*w + blockSize - 1) / blockSize;
   
   mult_scalar<<<numBlocks, blockSize>>>(h*w,scalar_value,matrix->d_rows);

	return 1;
}