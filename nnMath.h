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

double *allocate_vec_arr(int len);
double *allocate_mat_arr(int rows, int cols);

struct matrix *allocate_mat();
struct vector *allocate_vec();

double sigmoid(double val);
struct matrix sigmoid_matrix(struct matrix *mat);
struct matrix dsigmoid(struct matrix* mat);
struct vector sigmoid_vector(struct vector* vec);
struct vector dsigmoid_vector(struct vector* vec);