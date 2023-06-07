#ifndef NURAL_NET
#define NEURAL_NET

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nnMath.h"

typedef struct
{
    matrix_t *random_weights;
    vector_t *random_bias;
    vector_t nodes;
    vector_t activation;
    int output;
    int input;
} layer_t;

typedef struct
{
    layer_t *layers;
    layer_t input;
    layer_t output;
    int num_layers;
} neural_net_t;

void init_layer(layer_t *, int, int, matrix_t *, vector_t *, vector_t*, vector_t*);

void free_layer(matrix_t*, vector_t*, vector_t*, vector_t*);

void forward(vector_t *result, layer_t *input);

void activation(layer_t *input, int length);

void loss(vector_t *, vector_t*);

#endif