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
};

struct neuralnet
{
    struct layer *layers;
    struct layer input;
    struct layer output;
    int num_layers;
};

void init_layer(struct layer *, int, int, struct matrix *, struct vector *);

void init_bias(struct vector *, int);

void init_random(struct matrix *, int, int);

void free_weight(struct matrix*);

void free_bias(struct vector*);

void free_layer(struct matrix*, struct vector*);

//activation function
double sigmoid(double);

double dSigmoid(double);
