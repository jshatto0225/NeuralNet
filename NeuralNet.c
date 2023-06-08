#include "NeuralNet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

neural_net_t allocate_neural_net(int layers, int* layer_sizes)
{
    neural_net_t new_net;
    new_net.num_layers = layers;

    new_net.layers = (layer_t *)malloc(sizeof(layer_t) * layers);

    for (int i = 0; i < new_net.num_layers; i++)
    {
        if(i == 0)
            new_net.layers[i] = init_layer(layer_sizes[i], 1);
        else
            new_net.layers[i] = init_layer(layer_sizes[i], layer_sizes[i - 1]);
    }

    return new_net;
}

void free_network(neural_net_t *network)
{
    for(int i = 0; i < network->num_layers; i++)
    {
        free_layer(&network->layers[i]);
    }
    free(network->layers);
}

void free_layer(layer_t* layer)
{
    free_matrix(&layer->weights);
    free_vector(&layer->biases);
    free_vector(&layer->weighted_outputs);
    free_vector(&layer->activated_outputs);
    free_vector(&layer->error);
}

layer_t init_layer(int length, int previous_layer_length)
{
    layer_t out;
    out.length = length;

    //defining the rows and columns of random-weight matrix
    out.weights = init_matrix(length, previous_layer_length);

    for (int i = 0; i < out.weights.row; i++)
    {
        for (int j = 0; j < out.weights.col; j++)
        {
            //random vals between [0,1)
            out.weights.arr[j + i * out.weights.col] = (double)((double)rand() / (RAND_MAX / 2)) - 1;
        }
    }

    //init the size of vector
    out.biases = init_vector(length);
    
    //allocating random doubles to bias
    for (int i = 0; i < out.biases.len; i++)
    {   
        //random biases from [0, 1)
        out.biases.arr[i] = (double)(rand() / (RAND_MAX / 2)) - 1;
    }

    out.activated_outputs = init_vector(length);
    out.weighted_outputs = init_vector(length);

    out.error = init_vector(length);

    return out;
}

void feed_forward(layer_t *current_layer, layer_t *previous_layer)
{
    vector_t weight_inputs = init_vector(current_layer->weights.row);

    //example for second layer, [16x10][10x1]+[16x1]
    multiply_mat_vec(&weight_inputs, &current_layer->weights, &previous_layer->activated_outputs);

    add_vec(&current_layer->weighted_outputs, &weight_inputs, &current_layer->biases);

    free_vector(&weight_inputs);
}

void forward_pass(neural_net_t *network)
{
    for (int i = 1; i < network->num_layers; i++)
    {
        feed_forward(&network->layers[i], &network->layers[i - 1]);
        sigmoid_vec(&network->layers[i].activated_outputs, &network->layers[i].weighted_outputs);
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

void backward_pass(neural_net_t *network, vector_t *expected_outputs)
{
    for (int i = network->num_layers - 1; i > 0; i--)
    {
        if (i == network->num_layers - 1)
        {
            vector_t cost_derivative = init_vector(network->layers[i].length);
            subtract_vec(&cost_derivative, &network->layers[i].activated_outputs, expected_outputs);
            vector_t sigmoid_derivative = init_vector(network->layers[i].length);
            dsigmoid_vec(&sigmoid_derivative, &network->layers[i].weighted_outputs);

            hadamard_product(&network->layers[i].error, &cost_derivative, &sigmoid_derivative);

            free_vector(&cost_derivative);
            free_vector(&sigmoid_derivative);
        }
        else
        {
            matrix_t weights_transpose = init_matrix(network->layers[i + 1].weights.col, network->layers[i + 1].weights.row);
            transpose(&weights_transpose, &network->layers[i + 1].weights);
            vector_t error_product = init_vector(weights_transpose.row);
            multiply_mat_vec(&error_product, &weights_transpose, &network->layers[i + 1].error);

            vector_t sigmoid_derivative = init_vector(network->layers[i].length);
            dsigmoid_vec(&sigmoid_derivative, &network->layers[i].weighted_outputs);

            hadamard_product(&network->layers[i].error, &error_product, &sigmoid_derivative);

            free_matrix(&weights_transpose);
            free_vector(&error_product);
            free_vector(&sigmoid_derivative);
        }
    }
}

void train(neural_net_t *network, matrix_t *inputs, matrix_t *expected_outputs, 
           int epochs, int batch_size, double learning_rate,
           matrix_t *test_inputs, matrix_t *test_expected_outputs)
{
    for (int i = 0; i < epochs; i++)
    {
        printf("Starting epoch %d\n", i);
        // TODO: Implement batch training
        for (int j = 0; j < inputs->row; j++)
        {
            for (int k = 0; k < inputs->col; k++)
            {
                network->layers[0].activated_outputs.arr[k] = inputs->arr[j * inputs->col + k];
            }
            vector_t expected_outputs2 = init_vector(10);
            for (int i = 0; i < 10; i++)
            {
                expected_outputs2.arr[i] = expected_outputs->arr[j * expected_outputs->col + i];
            }
            forward_pass(network);
            backward_pass(network, &expected_outputs2);
            update_weights(network, learning_rate);
        }
    }
    test(network, test_inputs, test_expected_outputs);
}

void update_weights(neural_net_t *network, double learning_rate)
{
    for (int i = network->num_layers - 1; i > 0; i--)
    {
        matrix_t weights_product = init_matrix(network->layers[i].error.len, network->layers[i - 1].activated_outputs.len);
        multiply_vec_vec(&weights_product, &network->layers[i].error, &network->layers[i - 1].activated_outputs);
        scalar_multiply_mat(&weights_product, &weights_product, learning_rate);
        subtract_mat(&network->layers[i].weights, &network->layers[i].weights, &weights_product);

        free_matrix(&weights_product);

        vector_t bias_product = init_vector(network->layers[i].error.len);
        scalar_multiply_vec(&bias_product, &network->layers[i].error, learning_rate);
        subtract_vec(&network->layers[i].biases, &network->layers[i].biases, &bias_product);

        free_vector(&bias_product);
    }
}

void test(neural_net_t *network, matrix_t *inputs, matrix_t *expected_outputs)
{
    matrix_t network_output = init_matrix(inputs->row, inputs->col);
    int sum = 0;
    for (int i = 0; i < inputs->row; i++)
    {
        for (int j = 0; j < inputs->col; j++)
        {
            network->layers[0].activated_outputs.arr[j] = inputs->arr[i * inputs->col + j];
        }
        forward_pass(network);
        int max_index = 0;
        for (int j = 0; j < network->layers[network->num_layers - 1].activated_outputs.len; j++)
        {
            if (network->layers[network->num_layers - 1].activated_outputs.arr[j] > network->layers[network->num_layers - 1].activated_outputs.arr[max_index])
            {
                max_index = j;
            }
        }
        if(expected_outputs->arr[i * expected_outputs->col + max_index] == 1)
        {
            sum++;
        }
    }
    printf("Accuracy: %d / 10000\n", sum);
}

void print_matrix(matrix_t *mat)
{
    for (int i = 0; i < mat->row; i++)
    {
        for (int j = 0; j < mat->col; j++)
        {
            printf("%lf, ", mat->arr[j + i * mat->col]);
        }
        printf("\n");
    }
}
void print_vector(vector_t *vec)
{
    for (int i = 0; i < vec->len; i++)
    {
        printf("%lf\n", vec->arr[i]);
    }
}
