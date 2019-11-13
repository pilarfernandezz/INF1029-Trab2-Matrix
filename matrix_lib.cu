#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "matrix_lib.h"
#define THREADS_PER_BLOCK 256

// Kernel function to add the elements of two arrays
__global__ 
void mult_scalar(int n , float scalar, float *d_a){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x *blockDim.x;
    for(int i = index; i < n; i += stride){
       d_a[i] = d_a[i] * scalar;
    }
}

__global__
void mult_matrix(int w_a, int w_b, int h_b, int h_a, float *d_a, float *d_b, float *d_c){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x *blockDim.x;

    int w_c = h_a, h_c = w_b;
    // Calculando a matriz resultante
    for(int i = index; i < w_c*h_c; i += stride) {
        d_c[i] = 0;
        for(int j = 0; j < h_a; j++) {
            d_c[i] += d_a[(i/h_c)*h_a + j] * d_b[j*h_b + i%h_c];
        }
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

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC){	

	long unsigned int h_a;
	long unsigned int w_a;
	long unsigned int h_b;
	long unsigned int w_b;
	long unsigned int h_c;
	long unsigned int w_c;

    if(matrixA == NULL || matrixB == NULL|| matrixC == NULL) return 0;

    h_a = matrixA->height;
	w_a = matrixA->width;
	h_b = matrixB->height;
	w_b = matrixB->width;
	h_c = matrixC->height;
	w_c = matrixC->width;

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (h_c*w_c + blockSize - 1) / blockSize;
    mult_matrix<<<numBlocks, blockSize>>>(w_a, w_b, h_b, h_a,matrixA->d_rows, matrixB->d_rows, matrixC->d_rows);

    return 1;
}