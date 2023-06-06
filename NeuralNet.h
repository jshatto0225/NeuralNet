#ifndef NURAL_NET
#define NEURAL_NET

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nnMath.h"

struct layer
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
    struct layer *layers;
    struct layer input;
    struct layer output;
    int num_layers;
};

void allocate_neural_net(int, int*, struct layer**);

void init_layer(struct layer *, int, int, struct matrix *, struct vector *, struct vector*, struct vector*);

void free_network(int, struct layer**);

void forward(struct vector *result, struct layer *input);

void activation(struct layer *input, int length);

#endif