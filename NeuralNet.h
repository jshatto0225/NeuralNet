#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <nnMath.h>

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
