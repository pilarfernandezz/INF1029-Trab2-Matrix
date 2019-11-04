#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "exec_time.h"
#include <math.h>
#include "matrix_lib.h"
int main_func(int argc, char **argv)
{
    float scalar = atof(argv[1]);
    int count;
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

    count=0;
	float* vetA= (float*)malloc((matA.height*matA.width) * sizeof(float));
	float vetA_aux;

	for(int i=0; i<matA.height*matA.width; i++){
		fread((void*)(&vetA_aux), sizeof(vetA_aux), 1, file1);
		vetA[count]=vetA_aux;
		count++;
	}
	matA.h_rows = vetA;
	
	count=0;
	float* vetB= (float*)malloc((matB.height*matB.width) * sizeof(float));
	float vetB_aux;
	
	for(int i=0; i<matB.height*matB.width; i++){
		fread((void*)(&vetB_aux), sizeof(vetB_aux), 1, file2);
		vetB[count]=vetB_aux;
		count++;
	}
	matB.h_rows = vetB;
	
	Matrix matC;
	matC.height = matA.height;
	matC.width = matB.width;

	float* vetC= (float*)malloc((matC.height*matC.width) * sizeof(float));
	
	for(int i=0; i< (matC.height*matC.width); i++){
		vetC[i] = 0;
	}
	matC.h_rows = vetC;

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

    gettimeofday(&stop, NULL);
    printf("%f ms\n", timedifference_msec(start, stop));

    // Run kernel on elements on the GPU
    printf("Running kernel on elements of matA.d_rows ...");
    gettimeofday(&start, NULL);

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

    // Check for errors (all values should be 3.0f)
    printf("Checking for processing errors...");
    gettimeofday(&start, NULL);

    float maxError = 0.0f;
    float diffError = 0.0f;
    for (i = 0; i < (matA.height*matA.width); i++) {
        maxError = (maxError > (diffError=fabs(matA.h_rows[i]-3.0f)))? maxError : diffError;
        //printf("%d -> %f\n", i, h_y[i]);
    }

    gettimeofday(&stop, NULL);
    printf("%f ms\n", timedifference_msec(start, stop));
    printf("Max error: %f\n", maxError);

    // Free memory
    printf("Freeing memory...");
    gettimeofday(&start, NULL);
    cudaFree(matA.d_rows);
    cudaFree(matB.d_rows);
    free(matA.h_rows);
    free(matB.h_rows);
    gettimeofday(&stop, NULL);
    printf("%f ms\n", timedifference_msec(start, stop));
    
    return 0;
}