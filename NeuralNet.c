#include "NeuralNet.h"

void init_random(struct matrix* weights_matrix, int input, int output)
{
    //first creating the matrix struct
    weights_matrix = (struct matrix*)malloc(sizeof(struct matrix));

    //allocating space for the weight matrix rows
    weights_matrix->arr = (double **)malloc(output * sizeof(double*));
    weights_matrix->col = input;
    weights_matrix->row = output;
    for (int i = 0; i < weights_matrix->row; i++)
    {
        //allocatiog space for cols
        weights_matrix->arr[i] = (double *)malloc(input * sizeof(double));
        for (int j = 0; j < weights_matrix->col; j++)
        {
            //creaing the random values from [0,1)
            weights_matrix->arr[i][j]= (double)(rand() / (RAND_MAX+ 1.0));
        }
    }
    
}

void init_bias(struct vector* bias, int length){
    //init the vector struct
    bias = (struct vector*)malloc(sizeof(struct vector));

    //init the size of vector
    bias->len = length;
    //init the bias vector
    bias->arr = (double *)malloc(sizeof(struct vector)*bias->len);
    
    //allocating random doubles to bias
    for (int i = 0; i < bias->len; i++)
    {   
        //random biases from [0, 1)
        bias->arr[i]=  (double)(rand() / (RAND_MAX+ 1.0));
    }

}

void init_layer(struct layer* layer, int input, int output, struct matrix *weights, struct vector *biases)
{
    //init the layer struct
    layer = (struct layer*)malloc(sizeof(struct layer));

    //init the inputs and outputs of layer
    layer->input = input;
    layer->output = output;

    //init the weights and bias of the layer
    

    //first creating the matrix struct
    weights = (struct matrix*)malloc(sizeof(struct matrix));

    //allocating space for the weight matrix rows
    weights->arr = (double **)malloc(output * sizeof(double*));
    weights->col = input;
    weights->row = output;
    for (int i = 0; i < weights->row; i++)
    {
        //allocatiog space for cols
        weights->arr[i] = (double *)malloc(input * sizeof(double));
        for (int j = 0; j < weights->col; j++)
        {
            //creaing the random values from [0,1)
            weights->arr[i][j]= (double)(rand() / (RAND_MAX+ 1.0));
        }
    }


    init_bias(biases, input);
    layer->random_weights = weights;
    layer->random_bias = biases;
}

void free_weight(struct matrix* weights)
{
    for (int i = 0; i < weights->row; i++)
    {
        free(weights->arr[i]);
    }

    free(weights->arr);
    free(weights);
}

void free_bias(struct vector* bias)
{
    free(bias->arr);
    free(bias);
}

void free_layer(struct layer* layer)
{
    free_weight(layer->random_weights);

    free_bias(layer->random_bias);

    free(layer);
}