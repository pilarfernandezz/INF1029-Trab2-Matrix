#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "exec_time.h"
#include <math.h>
#include "matrix_lib.h"

int print_matrix(struct matrix *matrix) {
  unsigned long int i;
  unsigned long int N;
  unsigned long int nxt_newLine;

  /* Check the numbers of the elements of the matrix */
  N = matrix->height * matrix->width;

  /* Check the integrity of the matrix */
  if (N == 0 || matrix->h_rows == NULL) return 0;

  /* Initialize new line controol */
  nxt_newLine = matrix->width - 1;

  /* Print matrix elements */
  for (i = 0; i < N; i++) {
     printf("%5.1f ", matrix->h_rows[i]);
     if (i == nxt_newLine) {
	printf("\n");
	nxt_newLine += matrix->width;
     }
  }

  return 1;
}

int load_matrix(struct matrix *matrix, FILE *filename) {
  unsigned long int i = 0;
  unsigned long int N = 0;

  /* Check the numbers of the elements of the matrix */
  N = matrix->height * matrix->width;

  /* Check the integrity of the matrix */
  if (N == 0 || matrix->h_rows == NULL) return 0;

  float *nxt = matrix->h_rows; 

  for ( i = 0;
	i < N; 
	i += 1) {
    fread(nxt, sizeof(float), 1, filename);
    matrix->h_rows[i] = *nxt;
  }

  return 1;
}

int main_func(int argc, char **argv)
{
    float scalar = atof(argv[1]);
    Matrix matA, matB;
    matA.height = atoi(argv[2]);
    matA.width = atoi(argv[3]);
    matB.height = atoi(argv[4]);
    matB.width =atoi(argv[5]);

    FILE *file1, *file2, *result1, *result2;
  	file1 = fopen(argv[6],"rb");
	file2 = fopen(argv[7],"rb");
	result1 = fopen(argv[8],"wb");
	result2 = fopen(argv[9],"wb");

    if(file1 == NULL || file2 ==NULL)
	{
		fprintf(stdout, ".dat failed to open.  exiting...\n");
		exit(1);
	}

    cudaError_t cudaError;
    int i;
    struct timeval start, stop;

    // Disable buffering entirely
    setbuf(stdout, NULL);

    // Allocating arrays on host
    printf("Allocating matA.h_rows and matB.h_rows on host...");
    gettimeofday(&start, NULL);

    matA.h_rows = (float*)malloc((matA.height*matA.width)*sizeof(float));
    matB.h_rows = (float*)malloc((matB.height*matB.width)*sizeof(float));

    // check malloc memory allocation
    if (matA.h_rows == NULL || matB.h_rows == NULL) { 
        printf("Error: malloc unable to allocate memory on host.");
            return 1;
    }

    gettimeofday(&stop, NULL);
    printf("%f ms\n", timedifference_msec(start, stop));

    // Allocating array on device
    printf("Allocating array matA.d_rows and matB.d_rows on device...");
    gettimeofday(&start, NULL);

    cudaError = cudaMalloc(&matA.d_rows, matA.height*matA.width*sizeof(float));

    // check cudaMalloc memory allocation
    if (cudaError != cudaSuccess) {
        printf("cudaMalloc matA.d_rows returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
            return 1;
    }

    cudaError = cudaMalloc(&matB.d_rows, matB.height*matB.width*sizeof(float));

    // check cudaMalloc memory allocation
    if (cudaError != cudaSuccess) {
        printf("cudaMalloc matB.d_rows returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
            return 1;
    }

    gettimeofday(&stop, NULL);
    printf("%f ms\n", timedifference_msec(start, stop));

    // Initialize host memory
    printf("Initializing array matA and matB on host...");
    gettimeofday(&start, NULL);

    load_matrix(&matA, file1);
    load_matrix(&matB, file2);
    
    gettimeofday(&stop, NULL);
    printf("%f ms\n", timedifference_msec(start, stop));

    // Allocating array on device
    printf("Allocating array matC.d_rows on device...");
    gettimeofday(&start, NULL);

	Matrix matC;
	matC.height = matA.height;
	matC.width = matB.width;

	matC.h_rows = (float*)malloc((matC.height*matC.width)*sizeof(float));

    // check malloc memory allocation
    if (matC.h_rows == NULL) { 
        printf("Error: malloc unable to allocate memory on host.");
            return 1;
    }

    cudaError = cudaMalloc(&matC.d_rows, matC.height*matC.width*sizeof(float));

    // check cudaMalloc memory allocation
    if (cudaError != cudaSuccess) {
        printf("cudaMalloc matC.d_rows returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
            return 1;
    }

    gettimeofday(&stop, NULL);
    printf("%f ms\n", timedifference_msec(start, stop));

    // Copy array from host to device
    printf("Copying arrays from host to device...");
    gettimeofday(&start, NULL);

    cudaError = cudaMemcpy(matA.d_rows, matA.h_rows, (matA.width*matA.height)*sizeof(float), cudaMemcpyHostToDevice);

    if (cudaError != cudaSuccess) {
        printf("cudaMemcpy matA.d_rows -> matA.h_rows returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
            return 1;
    }

    cudaError = cudaMemcpy(matB.d_rows, matB.h_rows, (matB.width*matB.height)*sizeof(float), cudaMemcpyHostToDevice);

    if (cudaError != cudaSuccess) {
        printf("cudaMemcpy matB.d_rows -> matB.h_rows returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
            return 1;
    }

    cudaError = cudaMemcpy(matC.d_rows, matC.h_rows, (matC.width*matC.height)*sizeof(float), cudaMemcpyHostToDevice);

    if (cudaError != cudaSuccess) {
        printf("cudaMemcpy matC.d_rows -> matC.h_rows returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
            return 1;
    }

    gettimeofday(&stop, NULL);
    printf("%f ms\n", timedifference_msec(start, stop));

    // Run kernel on elements on the GPU
    printf("Running kernel on elements of matA.d_rows ...");
    gettimeofday(&start, NULL);

    float valorAantes = matA.h_rows[0];
    scalar_matrix_mult(scalar, &matA);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    gettimeofday(&stop, NULL);
    printf("%f ms\n", timedifference_msec(start, stop));

    // Copy array from device to host
    printf("Copying array from device matA.d_rows to host matA.h_rows...");
    gettimeofday(&start, NULL);
    
    cudaError = cudaMemcpy(matA.h_rows,matA.d_rows, (matA.height*matA.width) * sizeof(float), cudaMemcpyDeviceToHost);

    if (cudaError != cudaSuccess)
    {
        printf("cudaMemcpy matA.d_rows -> matA.h_rows returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        return 1;
    }

    gettimeofday(&stop, NULL);
    printf("%f ms\n", timedifference_msec(start, stop));
    
    //Check for errors
    printf("Checking for processing errors for multiplication of matrix a and scalar...");
    gettimeofday(&start, NULL);

    float maxError = 0.0f;
    float diffError = 0.0f;
    
    for (i = 0; i < (matA.height*matA.width); i++) {
        maxError = (maxError > (diffError=fabs(matA.h_rows[0]-scalar*valorAantes)))? maxError : diffError;
        //printf("%d -> %f\n", i, matA.h_rows[i]);
    }

    gettimeofday(&stop, NULL);
    printf("%f ms\n", timedifference_msec(start, stop));
    printf("Max error: %f\n", maxError);

    //print matrixA
    print_matrix(&matA);
    
    if(matrix_matrix_mult(&matA, &matB, &matC) !=1) return 0;

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy array from device to host
    printf("Copying array from device matC.d_rows to host matC.h_rows...\n");
    gettimeofday(&start, NULL);
    
    cudaError = cudaMemcpy(matC.h_rows,matC.d_rows, (matC.height*matC.width) * sizeof(float), cudaMemcpyDeviceToHost);

    if (cudaError != cudaSuccess)
    {
        printf("cudaMemcpy matC.d_rows -> matC.h_rows returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        return 1;
    }

    print_matrix(&matC);

    // Free memory
    printf("Freeing memory...");
    gettimeofday(&start, NULL);
    cudaFree(matA.d_rows);
    cudaFree(matB.d_rows);
    free(matA.h_rows);
    free(matB.h_rows);
    gettimeofday(&stop, NULL);
    printf("%f ms\n", timedifference_msec(start, stop));
    
    for(int i=0; i<matA.height*matA.width; i++){	
 		fwrite((void*)(&matA.h_rows[i]), sizeof(matA.h_rows[i]), 1, result1);
	}
    for(int i=0; i<matC.height*matC.width; i++){	
 		fwrite((void*)(&matC.h_rows[i]), sizeof(matC.h_rows[i]), 1, result2);
	}

	fclose(file1);
	fclose(file2);
	fclose(result1);
	fclose(result2);
    return 0;
}