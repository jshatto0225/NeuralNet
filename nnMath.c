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

struct vector multiply(struct matrix *mat, struct vector *vec)
{
    struct vector result = init_vector(mat->row);

    for (int i = 0; i < mat->row; i++)
    {
        for (int j = 0; j < mat->col; j++)
        {
            result.arr[i] += mat->arr[j + i * mat->col] * vec->arr[j];
        }
    }
    
    return result;
}

struct vector add(struct vector *v1, struct vector *v2)
{
    struct vector result = init_vector(v1->len);

    for (int i = 0; i < result.len; i++)
    {
        result.arr[i] = v1->arr[i] + v2->arr[i];
    }

    return result;
}

struct vector subtract(struct vector *v1, struct vector *v2)
{
    struct vector result = init_vector(v1->len);

    for (int i = 0; i < result.len; i++)
    {
        result.arr[i] = v1->arr[i] - v2->arr[i];
    }

    return result;
}

double sigmoid(double val)
{
    return 1 / (1 + exp(-1 * val));
}

struct matrix sigmoid_matrix(struct matrix* mat)
{
    struct matrix result = init_matrix(mat->row, mat->col);

    for (int i = 0; i < mat->row; i++)
    {
        for (int j = 0; j < mat->col; j++)
        {
            result.arr[j + i * result.col] = sigmoid(mat->arr[j + i*mat->col]);
        }
    }
    return result;
}

struct matrix dsigmoid_matrix(struct matrix* mat)
{
    struct matrix result = init_matrix(mat->row, mat->col);

    for (int i = 0; i < mat->row; i++)
    {
        for (int j = 0; j < mat->col; j++)
        {
            double sig = sigmoid(mat->arr[j + i * mat->col]);
            result.arr[j + i * result.col] = sig * (1 - sig);
        }
    }
    return result;
}

struct vector sigmoid_vector(struct vector* vec)
{
    struct vector result = init_vector(vec->len);

    for (int i = 0; i < result.len; i++)
    {
        result.arr[i] = sigmoid(vec->arr[i]);
    }
    return result;
}

struct vector dsigmoid_vector(struct vector* vec)
{
    struct vector result = init_vector(vec->len);

    for (int i = 0; i < result.len; i++)
    {
        double sig = sigmoid(vec->arr[i]);
        result.arr[i] = sig * (1 - sig);
    }
    return result;
}

struct vector hadamard_product(struct vector *vec1, struct vector *vec2)
{
    struct vector result = init_vector(vec1->len);

    for (int i = 0; i < result.len; i++)
    {
        result.arr[i] = vec1->arr[i] * vec2->arr[i];
    }
    return result;
}

struct vector output_error(struct vector *expected_output, 
                            struct vector *last_layer_activations, 
                            struct vector *last_layer_weighted)
{
    struct vector llw_dsig = dsigmoid_vector(last_layer_weighted);
    struct vector error = subtract(last_layer_activations, expected_output);

    struct vector result = hadamard_product(&error, &llw_dsig);

    free_vector(&llw_dsig);
    free_vector(&error);

    return result;
}

struct vector layer_error(struct matrix *next_layer_weights,
                          struct vector *next_layer_error,
                          struct vector *current_layer_weighted)
{
    struct vector clw_dsig = dsigmoid_vector(current_layer_weighted);
    struct matrix nlwT = transpose(next_layer_weights);
    struct vector product = multiply(&nlwT, next_layer_error);

    struct vector result = hadamard_product(&product, &clw_dsig);

    free_vector(&clw_dsig);
    free_matrix(&nlwT);
    free_matrix(&product);

    return result;
}

struct matrix transpose(struct matrix* mat)
{
    struct matrix result = init_matrix(mat->col, mat->row);

    for (int i = 0; i < result.row; i++)
    {
        for (int j = 0; j < result.col; j++)
        {
            result.arr[j + i * result.col] = mat->arr[i + j * mat->col];
        }
    }

    return result;
}

// int main()
// {
//     struct vector vec;
//     vec.len = 4;
//     vec.arr = allocate_vec_arr(4);
//     struct matrix mat;
//     mat.row = 6;
//     mat.col = 4;
//     mat.arr = allocate_mat_arr(6, 4);
// int main()
// {
//     struct vector vec;
//     vec.len = 4;
//     vec.arr = allocate_vec_arr(4);
//     struct matrix mat;
//     mat.row = 6;
//     mat.col = 4;
//     mat.arr = allocate_mat_arr(6, 4);

//     for (int i = 0; i < vec.len; i++)
//     {
//         vec.arr[i] = i;
//     }

//     int k = 0;
//     for (int i = 0; i < mat.row; i++)
//     {
//         for (int j = 0; j < mat.col; j++)
//         {
//             mat.arr[j + i*mat.col] = k;
//             k++;
//         }
//     }

//     struct vector mult = multiply(&mat, &vec);

//     for (int i = 0; i < mult.len; i++)
//     {
//         printf("%lf\n", mult.arr[i]);
//     }
// }