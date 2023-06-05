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

// Allocates space of a matrix and a vector
struct vector init_vector(int len);
struct matrix init_matrix(int row, int col);
double *allocate_vec_arr(int len);
double *allocate_mat_arr(int rows, int cols);
struct matrix *allocate_mat();
struct vector *allocate_vec();

// Frees memory for matrices and vectors
void free_vector(struct vector *vec);
void free_matrix(struct matrix *mat);

// Multiply a matrix with a vector
void multiply(struct vector *result, struct matrix *mat, struct vector *vec);
// Add two vectors
void add(struct vector *result, struct vector *v1, struct vector *v2);
// Subtract two vectors
void subtract(struct vector *result, struct vector *v1, struct vector *v2);

// Sigmoid of a single value
double sigmoid(double val);

// Sigmoid of a matrix
void sigmoid_matrix(struct matrix *result, struct matrix *mat);
// Sigmoid derivative of a matrix
void dsigmoid(struct matrix *result, struct matrix *mat);

// Sigmoid of a vector
void sigmoid_vector(struct vector *result, struct vector *vec);
// Sigmoid derivative of a vector
void dsigmoid_vector(struct vector *result, struct vector *vec);

void hadamard_product(struct vector *result, struct vector *vec1, struct vector *vec2);

// Gets the error at the output layer
// last_layer_activations - output of last layer after sigmoid
// last_layer_weighted - output of last layer before sigmoid
void output_error(struct vector *result,
                  struct vector *expected_output,
                  struct vector *last_layer_activations,
                  struct vector *last_layer_weighted);

void transpose(struct matrix *result, struct matrix *mat);

// Gets the error of all layers except output
// next_layer_weights - weights of next layer
// next_layer_error - error of the next layer
// output of the current layer before sigmoid
void layer_error(struct vector *result,
                 struct matrix *next_layer_weights,
                 struct vector *next_layer_error,
                 struct vector *current_layer_weighted);

#endif