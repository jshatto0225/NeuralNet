#ifndef NURAL_NET
#define NEURAL_NET

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nnMath.h"

typedef struct
{
    struct matrix random_weights;
    struct vector random_bias;
    struct vector weighted_outputs;
    struct vector activated_outputs;
    struct vector error;
    int length;
};

typedef struct layer* network;

struct neuralnet
{
    layer_t *layers;
    layer_t input;
    layer_t output;
    int num_layers;
} neural_net_t;

void allocate_neural_net(int, int*, struct layer**);

void init_layer(struct layer *, int, int, struct matrix *, struct vector *, struct vector*, struct vector*);

void free_network(int, struct layer**);

void forward(vector_t *result, layer_t *input);

void activation(layer_t *input, int length);

void loss(vector_t *, vector_t*);

#endif