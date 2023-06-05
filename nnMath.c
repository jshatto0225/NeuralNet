#include "nnMath.h"

// Gradient descent
// w is a matrix if weights
// w' is the new weights
// b is a vector of biases
// b' is the new biases
//
// n is the learning rate
// m is the number of samples
//
// dC/dw is the derivative of the cost function with respect to the weights
// dC/db os the derivative of the cost function with respect to the biases 
//
// w' = w - (n/m) * sum(all samples) of (dC/dw)
// b' = b - (n/m) * sum(all samples) of (dC/db)

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
    struct vector result;
    result.len = mat->row;
    result.arr = allocate_vec_arr(result.len);

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
    struct vector result;
    result.len = v1->len;
    result.arr = allocate_vec_arr(result.len);

    for (int i = 0; i < result.len; i++)
    {
        result.arr[i] = v1->arr[i] + v2->arr[i];
    }

    return result;
}

double sigmoid(double val)
{
    return 1 / (1 + exp(-1 * val));
}

struct matrix sigmoid_matrix(struct matrix* mat)
{
    struct matrix result;
    result.row = mat->row;
    result.col = mat->col;
    result.arr = allocate_mat_arr(result.row, result.col);

    for (int i = 0; i < mat->row; i++)
    {
        for (int j = 0; j < mat->col; j++)
        {
            result.arr[j + i * result.col] = sigmoid(mat->arr[j + i*mat->col]);
        }
    }
}

struct matrix dsigmoid_matrix(struct matrix* mat)
{
    struct matrix result;
    result.row = mat->row;
    result.col = mat->col;
    result.arr = allocate_mat_arr(result.row, result.col);

    for (int i = 0; i < mat->row; i++)
    {
        for (int j = 0; j < mat->col; j++)
        {
            double sig = sigmoid(mat->arr[j + i * mat->col]);
            result.arr[j + i * result.col] = sig * (1 - sig);
        }
    }
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