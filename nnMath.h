#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifndef NN_MATH_H
#define NN_MATH_H

struct matrix
{
    double *arr;
    int row;
    int col;
};

struct vector
{
    double *arr;
    int len;
};

struct vector init_vector(int len);
struct matrix init_matrix(int row, int col);

void free_vector(struct vector *vec);
void free_matrix(struct matrix *mat);

// Multiply a matrix with a vector
struct vector multiply(struct matrix *mat, struct vector *vec);
// Add two vectors
struct vector add(struct vector *v1, struct vector *v2);
// Subtract two vectors
struct vector subtract(struct vector *v1, struct vector *v2);

double *allocate_vec_arr(int len);
double *allocate_mat_arr(int rows, int cols);

struct matrix *allocate_mat();
struct vector *allocate_vec();

// Sigmoid of a single value
double sigmoid(double val);

// Sigmoid of a matrix
struct matrix sigmoid_matrix(struct matrix *mat);
// Sigmoid derivative of a matrix
struct matrix dsigmoid(struct matrix* mat);

// Sigmoid of a vector
struct vector sigmoid_vector(struct vector* vec);
// Sigmoid derivative of a vector
struct vector dsigmoid_vector(struct vector* vec);

struct vector hadamard_product(struct vector *vec1, struct vector *vec2);

// Gets the error at the output layer
// last_layer_activations - output of last layer after sigmoid
// last_layer_weighted - output of last layer before sigmoid
struct vector output_error(struct vector *expected_output,
                           struct vector *last_layer_activations,
                           struct vector *last_layer_weighted);

struct matrix transpose(struct matrix* mat);

// Gets the error of all layers except output
// next_layer_weights - weights of next layer
// next_layer_error - error of the next layer
// output of the current layer before sigmoid
struct vector layer_error(struct matrix *next_layer_weights,
                          struct vector *next_layer_error,
                          struct vector *current_layer_weighted);

#endif