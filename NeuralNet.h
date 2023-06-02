#include <stdlib.h>
#include <stdio.h>
#include <math.h>

struct matrix
{
    double **arr;
    int row;
    int col;
};

struct vector
{
    double *arr;
    int len;
};

struct layer
{
    struct matrix *random_weights;
    struct vector *bias;
    int output;
    int input;
};

struct neuralnet
{
    struct layer *layers;
    struct layer *input;
    struct layer *output;
    int len;
};

void init_layer(struct layer, int, int, struct matrix, struct vector);

void init_neuralnet(int, struct layer, struct layer, struct layer, struct neuralnet);

void init_random(struct matrix);

matrix multiply(matrix *mat1, matrix *mat2);
vector multiply(matrix *mat, vector *vec);
vector add(vector *v1, vector *v2);
