#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct matrix{
	unsigned long int height; //num de linhas da matriz (32 bits)
	unsigned long int width; //num de colunas da matriz (32 bits)
	float *h_rows; 
    float *d_rows;
};

typedef struct matrix Matrix;
int scalar_matrix_mult(float scalar_value, struct matrix *matrix);
int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC);