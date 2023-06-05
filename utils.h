#include <stdio.h>
#include "nnMath.h"

void print_matrix(struct matrix *mat)
{
    for (int i = 0; i < mat->row; i++)
    {
        for (int j = 0; j < mat->col; j++)
        {
            printf("%.2lf, ", mat->arr[j + i * mat->col]);
        }
        printf("\n");
    }
}
void print_vector(struct vector *vec)
{
    for (int i = 0; i < vec->len; i++)
    {
        printf("%.2lf\n", vec->arr[i]);
    }
}