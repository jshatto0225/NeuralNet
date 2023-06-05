#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nnMath.h"

struct layer
{
    struct matrix *random_weights;
    struct vector *random_bias;
    struct vector nodes;
    struct vector activation;
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

void init_layer(struct layer *, int, int, struct matrix *, struct vector *, struct vector*);

void free_layer(struct matrix*, struct vector*);

void forward(struct vector *result, struct layer *input);

void activation(struct vector *result, struct layer *input);
