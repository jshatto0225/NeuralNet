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
    init_random(weights, input, output);
    init_bias(biases, input);
    layer->random_weights = weights;
    layer->random_bias = biases;

}