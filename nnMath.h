#include <stdio.h>
#include <string.h>
#include <stdlib.h>

struct matrix
{
    double *arr;
    int row;
    int col;
};

struct vector
{
    double *arr;
    int len;
};

struct vector multiply(struct matrix *mat, struct vector *vec);
struct vector add(struct vector *v1, struct vector *v2);

double *allocate_vec(int len);
double *allocate_mat(int rows, int cols);