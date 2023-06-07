#ifndef NURAL_NET
#define NEURAL_NET

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nnMath.h"

typedef struct
{
    matrix_t random_weights;
    vector_t random_bias;
    vector_t weighted_outputs;
    vector_t activated_outputs;
    vector_t error;
    int length;
} layer_t;

typedef layer_t* network;

typedef struct neuralnet
{
    layer_t *layers;
    layer_t input;
    layer_t output;
    int num_layers;
} neural_net_t;

void allocate_neural_net(int, int*, layer_t**);

void init_layer(layer_t *, int, int, matrix_t *, vector_t *, vector_t*, vector_t*);

void free_network(int, layer_t**);

void forward(vector_t *result, layer_t *input);

void activation(layer_t *input, int length);

void loss(vector_t *, vector_t*);

#endif