#include "NeuralNet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void init_layer(layer_t* layer, int input, int output, matrix_t *weights, vector_t *biases, vector_t *nodes, vector_t *active)
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
            //random vals between [0,1)
            weights->arr[j + i * weights->col] = (double)(rand() / (RAND_MAX+ 1.0));
        }
    }

    //init the size of vector
    biases->len = output;

    //if its the last layer
    if (output == 0)
    {
        biases->len = input;
    }
    //init the bias vector
    biases->arr = allocate_vec_arr(biases->len);
    
    //allocating random doubles to bias
    for (int i = 0; i < biases->len; i++)
    {   
        //random biases from [0, 1)
        biases->arr[i] = (double)(rand() / (RAND_MAX+ 1.0));
    }
    
    //allocating space for nodes

    nodes->len = input;
    nodes->arr = calloc(input, sizeof(double));

    active->len = input;
    active->arr = calloc(input, sizeof(double));


    layer->nodes = *nodes;
    layer->activation = *active;
    layer->random_weights = weights;
    layer->random_bias = biases;
}


void free_layer(matrix_t* matrix, vector_t* vector, vector_t* node, vector_t* active)
{
    free(matrix->arr);
    free(vector->arr);
    free(node->arr);
    free(active->arr);
}

void forward(vector_t *result, layer_t *input)
{
    // TODO: find length of weight_inputs
    vector_t weight_inputs;
    weight_inputs.len = input->random_weights->row;
    weight_inputs.arr = allocate_vec_arr(input->random_weights->row);

    //example for second layer, [16x10][10x1]+[16x1]
    multiply(&weight_inputs, input->random_weights, &input->nodes);

    add(result, &weight_inputs, input->random_bias);

    free_vector(&weight_inputs);
}

void activation(layer_t *input, int length)
{
    int i = 0;
    while (i < length)
    {
        //make sure its not the first layer bc it already has iputs
        if(i != 0)
        {
            forward(&input[i].nodes, &input[i-1]);
            sigmoid_vector(&input[i].activation, &input[i].nodes);
        }
        i++;
    }
}

double loss_function(vector_t *predict, vector_t *actual)
{
    double sum = 0;
    for (int i = 0; i < predict->len; i++)
    {
        sum = fabs(predict->arr[i] - actual->arr[i]);
    }
    return sum / predict->len;
}

