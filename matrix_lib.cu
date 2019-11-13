#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "matrix_lib.h"
#define THREADS_PER_BLOCK 256
//Função kernel para multiplicar matriz A por um número escalar, atualizando o valor da matriz A
__global__ 
void mult_scalar(int n , float scalar, float *d_a){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x *blockDim.x;
    for(int i = index; i < n; i += stride){
       d_a[i] = d_a[i] * scalar;
    }
}

//Função kernel para multiplicar a matriz A pela matriz B, guardando o resultado em uma matriz C
__global__
void mult_matrix(int w_a, int w_b, int h_b, int h_a, float *d_a, float *d_b, float *d_c){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x *blockDim.x;

    int w_c = h_a;
    int h_c = w_b;

    // Calculando a matriz resultante
    for(int i = index; i < w_c*h_c; i += stride) {
        d_c[i] = 0;
        for(int j = 0; j < w_a; j++) {
            d_c[i] += d_a[(i/w_c)*w_a + j] * d_b[w_b*j + i%w_c];
        }
    }
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix){
	if(matrix == NULL) return 0;

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (matrix->height*matrix->width + blockSize - 1) / blockSize;
    mult_scalar<<<numBlocks, blockSize>>>(matrix->height*matrix->width,scalar_value,matrix->d_rows);

	return 1;
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC){	

    if(matrixA == NULL || matrixB == NULL|| matrixC == NULL) return 0;      
    
    if(matrixA->width != matrixB->height){
        printf("Não é possível multiplicar matriz %dx%d por outra %dx%d\n", matrixA->height, matrixA->width, matrixB->height, matrixB->width);
        return 0;
    }

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (matrixC->height*matrixC->width + blockSize - 1) / blockSize;
    mult_matrix<<<numBlocks, blockSize>>>(matrixA->width, matrixB->width, matrixB->height, matrixA->height ,matrixA->d_rows, matrixB->d_rows, matrixC->d_rows);
    
    return 1;
}