#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nnMath.h"

struct layer
{
    struct matrix *random_weights;
    struct vector *random_bias;
    int output;
    int input;
    double * nodes;
};

struct neuralnet
{
    struct layer *layers;
    struct layer input;
    struct layer output;
    int num_layers;
};

void init_layer(struct layer *, int, int, struct matrix *, struct vector *, double*);

void free_layer(struct matrix*, struct vector*);

double forward(double* , struct layer);
