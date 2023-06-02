#include "NeuralNet.h"

void init_random(struct matrix* weights_matrix, int input, int output)
{
    //first creating the weight matrix struct
    weights_matrix = (struct matrix*)malloc(sizeof(struct matrix));

    //allocating space for the weight matrix rows
    weights_matrix->arr = (double **)malloc(output * sizeof(double*));
    for (int i = 0; i < output; i++)
    {
        //allocatiog space for cols
        weights_matrix->arr[i] = (double *)malloc(input * sizeof(double));
        for (int j = 0; j < input; j++)
        {
            //creaing the random values from [0,1)
            weights_matrix->arr[i][j]= (double)(rand() / (RAND_MAX+ 1.0));
        }
    }
    
}