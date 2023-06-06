#include "nnMath.h"
#include "utils.h"

// * is the hadamard product
// L is the last layer
// y is a vector of outputs
// z is the weighted output of a layer
// a is the activated output of a layer
// w are the weights of a layer
// b are the biases of a layer
// dsigmoid is the derivative of the sigmoud function
// delta is the error of a layer
// ^T is the transpose of a matrix
// l is a layer index
// n is the learning rate
// m is the number of samples
// sum is a summation over all of the samples x
//
// For each training example x:
//  Feedforward:
//      For each layer l:
//          z(x, l) = w(l)a(x, l-1) + b(l)
//          a(x, l) = sigmoid(z(x, l))
//  Output Error:
//      delta(x, L) = (a(L) - y) * dsigmoid(z(x, L))

//  Backpropogate Error:
//      For each layer l starting at L:
//          delta(x, l) = (w(l+1)^T delta(x, l+1)) * dsigmoid(z(x, l))
//
// Gradient descent
//  For each layer l starting at L:
//      w(l) = w(l) - (n/m)sum(delta(x, l)a(x, l-1)^T)
//      b(l) = b(l) -   (n/m)sum(delta(x, l))

struct vector init_vector(int len)
{
    struct vector vec;
    vec.len = len;
    vec.arr = allocate_vec_arr(vec.len);
}
struct matrix init_matrix(int row, int col)
{
    struct matrix mat;
    mat.row = row;
    mat.col = col;
    mat.arr = allocate_mat_arr(mat.row, mat.col);
}

void free_vector(struct vector *vec)
{
    free(vec->arr);
}
void free_matrix(struct matrix *mat)
{
    free(mat->arr);
}

double *allocate_mat_arr(int row, int col)
{
    return calloc(row*col, sizeof(double));
}
double *allocate_vec_arr(int len)
{
    return calloc(len, sizeof(double));
}

struct matrix *allocate_mat()
{
    return malloc(sizeof(struct matrix));
}

struct vector*allocate_vec()
{
    return malloc(sizeof(struct vector));
}

void multiply(struct vector *result, struct matrix *mat, struct vector *vec)
{
    for (int i = 0; i < mat->row; i++)
    {
        for (int j = 0; j < mat->col; j++)
        {
            result->arr[i] += mat->arr[j + i * mat->col] * vec->arr[j];
        }
    }
}

void add(struct vector *result, struct vector *v1, struct vector *v2)
{
    for (int i = 0; i < result->len; i++)
    {
        result->arr[i] = v1->arr[i] + v2->arr[i];
    }
}

void subtract(struct vector *result, struct vector *v1, struct vector *v2)
{
    for (int i = 0; i < result->len; i++)
    {
        result->arr[i] = v1->arr[i] - v2->arr[i];
    }
}

double sigmoid(double val)
{
    return 1 / (1 + exp(-1 * val));
}

void sigmoid_matrix(struct matrix *result, struct matrix* mat)
{
    for (int i = 0; i < mat->row; i++)
    {
        for (int j = 0; j < mat->col; j++)
        {
            result->arr[j + i * result->col] = sigmoid(mat->arr[j + i*mat->col]);
        }
    }
}

void dsigmoid_matrix(struct matrix *result, struct matrix* mat)
{
    for (int i = 0; i < mat->row; i++)
    {
        for (int j = 0; j < mat->col; j++)
        {
            double sig = sigmoid(mat->arr[j + i * mat->col]);
            result->arr[j + i * result->col] = sig * (1 - sig);
        }
    }
}

void sigmoid_vector(struct vector *result, struct vector* vec)
{
    for (int i = 0; i < result->len; i++)
    {
        result->arr[i] = sigmoid(vec->arr[i]);
    }
}

void dsigmoid_vector(struct vector *result, struct vector* vec)
{
    for (int i = 0; i < result->len; i++)
    {
        double sig = sigmoid(vec->arr[i]);
        result->arr[i] = sig * (1 - sig);
    }
}

void hadamard_product(struct vector *result, struct vector *vec1, struct vector *vec2)
{
    for (int i = 0; i < result->len; i++)
    {
        result->arr[i] = vec1->arr[i] * vec2->arr[i];
    }
}

void output_error(struct vector *result,
                  struct vector *expected_output, 
                  struct vector *last_layer_activations, 
                  struct vector *last_layer_weighted)
{
    struct vector llw_dsig = init_vector(last_layer_weighted->len);
    dsigmoid_vector(&llw_dsig, last_layer_weighted);

    struct vector error = init_vector(last_layer_activations->len);
    subtract(&error, last_layer_activations, expected_output);

    hadamard_product(result, &error, &llw_dsig);

    free_vector(&llw_dsig);
    free_vector(&error);
}

void layer_error(struct vector *result, 
                          struct matrix *next_layer_weights,
                          struct vector *next_layer_error,
                          struct vector *current_layer_weighted)
{
    struct vector clw_dsig = init_vector(current_layer_weighted->len);
    dsigmoid_vector(&clw_dsig, current_layer_weighted);

    struct matrix nlwT = init_matrix(next_layer_weights->col, next_layer_weights->row);
    transpose(&nlwT, next_layer_weights);

    struct vector product = init_vector(nlwT.row);
    multiply(&product, &nlwT, next_layer_error);

    hadamard_product(result, &product, &clw_dsig);

    free_vector(&clw_dsig);
    free_matrix(&nlwT);
    free_vector(&product);
}

void transpose(struct matrix *result, struct matrix* mat)
{
    for (int i = 0; i < result->row; i++)
    {
        for (int j = 0; j < result->col; j++)
        {
            result->arr[j + i * result->col] = mat->arr[i + j * mat->col];
        }
    }
}
