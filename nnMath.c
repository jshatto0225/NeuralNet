#include "nnMath.h"

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

int main()
{
    struct vector vec;
    vec.len = 4;
    vec.arr = allocate_vec_arr(4);
    struct matrix mat;
    mat.row = 6;
    mat.col = 4;
    mat.arr = allocate_mat_arr(6, 4);

    for (int i = 0; i < vec.len; i++)
    {
        vec.arr[i] = i;
    }

    int k = 0;
    for (int i = 0; i < mat.row; i++)
    {
        for (int j = 0; j < mat.col; j++)
        {
            mat.arr[j + i*mat.col] = k;
            k++;
        }
    }

    struct vector mult = multiply(&mat, &vec);

    for (int i = 0; i < mult.len; i++)
    {
        printf("%lf\n", mult.arr[i]);
    }
}