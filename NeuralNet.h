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
    struct vector *random_bias;
    int output;
    int input;
};

struct neuralnet
{
    struct layer *layers;
    struct layer input;
    struct layer output;
    int num_layers;
};

void init_layer(struct layer* , int, int, struct matrix*, struct vector*);

void init_bias(struct vector*, int);

void init_random(struct matrix*, int, int);

struct matrix multiply(struct matrix *mat1, struct matrix *mat2);
struct vector multiply(struct matrix *mat, struct vector *vec);
struct vector add(struct vector *v1, struct vector *v2);
