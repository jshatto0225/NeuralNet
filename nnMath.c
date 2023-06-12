#include "nnMath.h"
#include <omp.h>

vector_t init_vector(int len)
{
    vector_t vec;
    vec.len = len;
    vec.arr = allocate_vec_arr(vec.len);

    return vec;
}
matrix_t init_matrix(int row, int col)
{
    matrix_t mat;
    mat.row = row;
    mat.col = col;
    mat.arr = allocate_mat_arr(mat.row, mat.col);

    return mat;
}

void free_vector(vector_t *vec)
{
    free(vec->arr);
}
void free_matrix(matrix_t *mat)
{
    free(mat->arr);
}

double *allocate_mat_arr(int row, int col)
{
    return (double *)calloc(row * col, sizeof(double));
}
double *allocate_vec_arr(int len)
{
    return (double *)calloc(len, sizeof(double));
}

void multiply_mat_vec(vector_t *out, matrix_t *mat, vector_t *vec)
{
#pragma omp parallel for
    for (int i = 0; i < mat->row; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < mat->col; j++)
        {
            out->arr[i] += mat->arr[j + i * mat->col] * vec->arr[j];
        }
    }
}

void add_vec(vector_t *out, vector_t *v1, vector_t *v2)
{
#pragma omp parallel for
    for (int i = 0; i < out->len; i++)
    {
        out->arr[i] = v1->arr[i] + v2->arr[i];
    }
}

void subtract_vec(vector_t *out, vector_t *v1, vector_t *v2)
{
#pragma omp parallel for
    for (int i = 0; i < out->len; i++)
    {
        out->arr[i] = v1->arr[i] - v2->arr[i];
    }
}

double sigmoid(double val)
{
    return 1 / (1 + exp(-1 * val));
}

void sigmoid_mat(matrix_t *out, matrix_t *mat)
{
#pragma omp parallel for
    for (int i = 0; i < mat->row; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < mat->col; j++)
        {
            out->arr[j + i * out->col] = sigmoid(mat->arr[j + i * mat->col]);
        }
    }
}

void dsigmoid_mat(matrix_t *out, matrix_t *mat)
{
#pragma omp parallel for
    for (int i = 0; i < mat->row; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < mat->col; j++)
        {
            double sig = sigmoid(mat->arr[j + i * mat->col]);
            out->arr[j + i * out->col] = sig * (1 - sig);
        }
    }
}

void sigmoid_vec(vector_t *out, vector_t *vec)
{
#pragma omp parallel for
    for (int i = 0; i < out->len; i++)
    {
        out->arr[i] = sigmoid(vec->arr[i]);
    }
}

void dsigmoid_vec(vector_t *out, vector_t *vec)
{
#pragma omp parallel for
    for (int i = 0; i < out->len; i++)
    {
        double sig = sigmoid(vec->arr[i]);
        out->arr[i] = sig * (1 - sig);
    }
}

void hadamard_product(vector_t *out, vector_t *vec1, vector_t *vec2)
{
#pragma omp parallel for
    for (int i = 0; i < out->len; i++)
    {
        out->arr[i] = vec1->arr[i] * vec2->arr[i];
    }
}

void transpose(matrix_t *out, matrix_t *mat)
{
#pragma omp parallel for
    for (int i = 0; i < out->row; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < out->col; j++)
        {
            out->arr[j + i * out->col] = mat->arr[i + j * mat->col];
        }
    }
}

void multiply_vec_vec(matrix_t *out, vector_t *v1, vector_t *v2)
{
#pragma omp parallel for
    for (int i = 0; i < out->row; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < out->col; j++)
        {
            out->arr[j + i * out->col] = v1->arr[i] * v2->arr[j];
        }
    }
}

void scalar_multiply_mat(matrix_t *out, matrix_t *mat, double scalar)
{
#pragma omp parallel for
    for (int i = 0; i < out->row; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < out->col; j++)
        {
            out->arr[j + i * out->col] = mat->arr[j + i * mat->col] * scalar;
        }
    }
}

void subtract_mat(matrix_t *out, matrix_t *mat1, matrix_t *mat2)
{
#pragma omp parallel for
    for (int i = 0; i < out->row; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < out->col; j++)
        {
            out->arr[j + i * out->col] = mat1->arr[j + i * mat1->col] - mat2->arr[j + i * mat2->col];
        }
    }
}

void scalar_multiply_vec(vector_t *out, vector_t *vec, double scalar)
{
#pragma omp parallel for
    for (int i = 0; i < out->len; i++)
    {
        out->arr[i] = vec->arr[i] * scalar;
    }
}

void add_mat(matrix_t *out, matrix_t *mat1, matrix_t *mat2)
{
#pragma omp parallel for
    for (int i = 0; i < out->row; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < out->col; j++)
        {
            out->arr[j + i * out->col] = mat1->arr[j + i * mat1->col] + mat2->arr[j + i * mat2->col];
        }
    }
}

double ReLU(double val)
{
    return val > 0 ? val : 0;
}

void ReLU_mat(matrix_t *out, matrix_t *mat)
{
#pragma omp parallel for
    for (int i = 0; i < mat->row; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < mat->col; j++)
        {
            out->arr[j + i * out->col] = ReLU(mat->arr[j + i * mat->col]);
        }
    }
}

void dReLU(matrix_t *out, matrix_t *mat)
{
#pragma omp parallel for
    for (int i = 0; i < mat->row; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < mat->col; j++)
        {
            out->arr[j + i * out->col] = mat->arr[j + i * mat->col] > 0 ? 1 : 0;
        }
    }
}

void ReLU_vec(vector_t *out, vector_t *vec)
{
#pragma omp parallel for
    for (int i = 0; i < vec->len; i++)
    {
        out->arr[i] = ReLU(vec->arr[i]);
    }
}

void dReLU_vec(vector_t *out, vector_t *vec)
{
#pragma omp parallel for
    for (int i = 0; i < vec->len; i++)
    {
        out->arr[i] = vec->arr[i] > 0 ? 1 : 0;
    }
}