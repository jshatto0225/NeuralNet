#include "NeuralNet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>



void init_layer(struct layer* layer, int input, int output, struct matrix *weights, struct vector *biases, double *nodes)
{

    //init the inputs and outputs of layer
    layer->input = input;
    layer->output = output;    

    //defining the rows and columns of random-weight matrix
    weights->col = input;
    weights->row = output;


    //allocating space for the weight matrix rows & cols
    weights->arr = allocate_mat_arr(input, output);

    for (int i = 0; i < weights->row; i++)
    {
        for (int j = 0; j < input; j++)
        {
            weights->arr[j + i * weights->col] = (double)(rand() / (RAND_MAX+ 1.0));
        }
    }

    //init the size of vector
    biases->len = input;
    //init the bias vector
    biases->arr = allocate_vec_arr(biases->len);
    
    //allocating random doubles to bias
    for (int i = 0; i < biases->len; i++)
    {   
        //random biases from [0, 1)
        biases->arr[i]=  (double)(rand() / (RAND_MAX+ 1.0));
    }



    layer->random_weights = weights;
    layer->random_bias = biases;
}


void free_layer(struct matrix* matrix, struct vector* vector)
{
     free(matrix->arr);
    free(vector->arr);

}

double forward(double *arr, struct layer input)
{

}
