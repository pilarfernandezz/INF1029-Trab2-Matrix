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

    // int l = 1;

    // for(int i = index; i < h_a*w_b; i += stride){
    //     for(int j = (l-1)*w_a; j < l*w_a;j++){
    //         for (int k = l-1; k < w_b; k+=w_b){
    //             d_c[i] += d_a[j] * d_b[k];
    //             printf("%f \n", d_a[j]);
    //             printf("%f \n", d_b[k]);
    //             printf("%f \n", d_c[i]);
    //         }
    //     }
    //     l++;
    // }
    // for(int k = index; k < h_b*w_a;k+=stride){
    //     for (int i = 0; i < w_a*h_a ; i++){
    //         for (int j = 0; j < w_b; j++){
    //             d_c[k] += d_a[i] *d_b[j];
    //         }
    //     }
    // }
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

    //if errado


    float **mat = (float **) malloc (w_a*sizeof(float*));
    for(int i = 0; i< w_a; i++) mat[i] = (float *) malloc (h_a*sizeof(float));

    
    memcpy(mat, &matrixA->d_rows,h_a*w_a*sizeof(float));

    for (int i = 0; i < w_a;i++){
        for(int j = 0; j < h_a;j++){
            printf("matriz[%d][%d] = %f", i,j,mat[i][j]);
        }
    }


    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (h_c*w_c + blockSize - 1) / blockSize;
    mult_matrix<<<numBlocks, blockSize>>>(w_a, w_b, h_b, h_a,matrixA->d_rows, matrixB->d_rows, matrixC->d_rows);

    return 1;
}