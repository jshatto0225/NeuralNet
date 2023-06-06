#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <stdlib.h>

double* x_train()
{
    double* x = (double*)calloc(40, sizeof(double));
    for (int i = 0; i < 40; i++)
    {
        x[i] = 0.02 * i;
    }
    return x;
}


double* x_test()
{
    double* x = (double*)calloc(40, sizeof(double));
    for (int i = 0; i < 10; i++)
    {
        x[i] = 0.02 * i+40;
    }
    return x;
}



double* y_train(double * x)
{
    double* y = (double*)calloc(40, sizeof(double));
    for (int i = 0; i < 40; i++)
    {
        y[i] = 0.7 * x[i] + 0.3;
    }
    return y;
}


double* y_test(double * x)
{
    double* y = (double*)calloc(40, sizeof(double));
    for (int i = 0; i < 10; i++)
    {
        y[i] = 0.7 * x[i+40] +0.3;
    }
    return y;
}
