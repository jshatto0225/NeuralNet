#include "NeuralNet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>



// void init_random(struct matrix* weights_matrix, int input, int output)
// {
//     //first creating the matrix struct
//     weights_matrix = (struct matrix*)malloc(sizeof(struct matrix));

//     //allocating space for the weight matrix rows
//     weights_matrix->arr = (double **)malloc(output * sizeof(double*));
//     weights_matrix->col = input;
//     weights_matrix->row = output;
//     for (int i = 0; i < weights_matrix->row; i++)
//     {
//         //allocatiog space for cols
//         weights_matrix->arr[i] = (double *)malloc(input * sizeof(double));
//         for (int j = 0; j < weights_matrix->col; j++)
//         {
//             //creaing the random values from [0,1)
//             weights_matrix->arr[i][j]= (double)(rand() / (RAND_MAX+ 1.0));
//         }
//     }
    
// }

// void init_bias(struct vector* bias, int length){
//     //init the vector struct
//     bias = (struct vector*)malloc(sizeof(struct vector));

//     //init the size of vector
//     bias->len = length;
//     //init the bias vector
//     bias->arr = (double *)malloc(sizeof(struct vector)*bias->len);
    
//     //allocating random doubles to bias
//     for (int i = 0; i < bias->len; i++)
//     {   
//         //random biases from [0, 1)
//         bias->arr[i]=  (double)(rand() / (RAND_MAX+ 1.0));
//     }

// }

void init_layer(struct layer* layer, int input, int output, struct matrix *weights, struct vector *biases)
{
    //init the layer struct

    //init the inputs and outputs of layer
    layer->input = input;
    layer->output = output;

    //init the weights and bias of the layer
    

    //defining the rows and columns of random-weight matrix
    weights->col = input;
    weights->row = output;

    //allocating space for matrix struct


    //allocating space for the weight matrix rows & cols
    weights->arr = allocate_mat_arr(input, output);
    //need tto free

    for (int i = 0; i < weights->row; i++)
    {
        for (int j = 0; j < input; j++)
        {
            weights->arr[j + i * weights->col] = (double)(rand() / (RAND_MAX+ 1.0));
        }
    }


    //init the vector struct

    //init the size of vector
    biases->len = input;
    //init the bias vector
    biases->arr = allocate_vec_arr(biases->len);
    //free this too 
    
    //allocating random doubles to bias
    for (int i = 0; i < biases->len; i++)
    {   
        //random biases from [0, 1)
        biases->arr[i]=  (double)(rand() / (RAND_MAX+ 1.0));
    }



    layer->random_weights = weights;
    layer->random_bias = biases;
}

// void free_weight(struct matrix* weights)
// {
//     free(weights->arr);
//     free(weights);
// }

// void free_bias(struct vector* bias)
// {
//     free(bias->arr);
//     free(bias);
// }

void free_layer(struct matrix* matrix, struct vector* vector)
{
     free(matrix->arr);
    free(vector->arr);

}
