#include "NeuralNet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void allocate_neural_net(int layers, int* layer_sizes, struct layer** network)
{
    *network = (struct layer*)malloc(sizeof(struct layer) * layers);


    for(int i = 0; i < layers; i++)
    {
        struct layer new_layer;

        if(i > 0)
        {
            new_layer.random_weights.row = layer_sizes[i];
            new_layer.random_weights.col = layer_sizes[i - 1];
            new_layer.random_weights.arr = (double*)malloc(sizeof(double) * new_layer.random_weights.row * new_layer.random_weights.col);

            for (int i = 0; i < new_layer.random_weights.row; i++)
            {
                for (int j = 0; j < new_layer.random_weights.col; j++)
                {
                    //random vals between [0,1)
                    new_layer.random_weights.arr[j + i * new_layer.random_weights.col] = (double)(rand() / (RAND_MAX+ 1.0));
                }
            }

        }
        
        new_layer.random_bias.len = layer_sizes[i];
        new_layer.random_bias.arr = (double*)malloc(sizeof(double) * new_layer.random_bias.len);
        for(int i = 0; i < new_layer.random_bias.len; i++)
        {
            new_layer.random_bias.arr[i] = (double)(rand() / (RAND_MAX+ 1.0));
        }

        new_layer.weighted_outputs.len = layer_sizes[i];
        new_layer.weighted_outputs.arr = (double*)malloc(sizeof(double) * new_layer.weighted_outputs.len);
        for(int i = 0; i < new_layer.weighted_outputs.len; i++)
        {
            new_layer.weighted_outputs.arr[i] = 0;
        }

        new_layer.activated_outputs.len = layer_sizes[i];
        new_layer.activated_outputs.arr = (double*)malloc(sizeof(double) * new_layer.activated_outputs.len);
        for(int i = 0; i < new_layer.activated_outputs.len; i++)
        {
            new_layer.activated_outputs.arr[i] = 0;
        }

        new_layer.error.len = layer_sizes[i];
        new_layer.error.arr = (double*)malloc(sizeof(double) * new_layer.error.len);
        for(int i = 0; i < new_layer.error.len; i++)
        {
            new_layer.error.arr[i] = 0;
        }

        (*network + i)->random_weights = new_layer.random_weights;
        (*network + i)->random_bias = new_layer.random_bias;
        (*network + i)->weighted_outputs = new_layer.weighted_outputs;
        (*network + i)->activated_outputs = new_layer.activated_outputs;
        (*network + i)->length = layer_sizes[i];
        (*network + i)->error = new_layer.error;

    }
}
/*


void init_layer(struct layer* layer, int input, int output, struct matrix *weights, struct vector *biases, struct vector *nodes, struct vector *active)
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

*/
void free_network(int layers, struct layer** network)
{
    for(int i = 0; i < layers; i++)
    {
        free((*network + i)->random_weights.arr);
        free((*network + i)->random_bias.arr);
        free((*network + i)->weighted_outputs.arr);
        free((*network + i)->activated_outputs.arr);
        free((*network + i)->error.arr);
    }
    free(*network);
}
/*
void forward(struct vector *result, struct layer* input)
{
    // TODO: find length of weight_inputs
    struct vector weight_inputs;
    weight_inputs.len = input->random_weights->row;
    weight_inputs.arr = allocate_vec_arr(input->random_weights->row);

    //example for second layer, [16x10][10x1]+[16x1]
    multiply(&weight_inputs, input->random_weights, &input->nodes);

    add(result, &weight_inputs, input->random_bias);

    free_vector(&weight_inputs);
}

void activation(struct layer *input, int length)
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


*/
double loss_function(struct vector predict, struct vector actual)
{
    double sum = 0;
    for (int i = 0; i < predict.len; i++)
    {
        sum = fabs(predict.arr[i] - actual.arr[i]);
    }
    return sum / predict.len;
}

