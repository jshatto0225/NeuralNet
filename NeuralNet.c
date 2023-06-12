#include "NeuralNet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

neural_net_t allocate_neural_net(int layers, int *layer_sizes)
{
    neural_net_t new_net;
    new_net.num_layers = layers;

    new_net.layers = (layer_t *)malloc(sizeof(layer_t) * layers);

    for (int i = 0; i < new_net.num_layers; i++)
    {
        if (i == 0)
            new_net.layers[i] = init_layer(layer_sizes[i], 1);
        else
            new_net.layers[i] = init_layer(layer_sizes[i], layer_sizes[i - 1]);
    }

    return new_net;
}

void free_network(neural_net_t *net)
{
    for (int i = 0; i < net->num_layers; i++)
    {
        free_layer(&net->layers[i]);
    }
    free(net->layers);
}

void free_layer(layer_t *layer)
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

    // defining the rows and columns of random-weight matrix
    out.weights = init_matrix(length, previous_layer_length);

#pragma omp parallel for
    for (int i = 0; i < out.weights.row; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < out.weights.col; j++)
        {
            // random vals between [0,1)
            out.weights.arr[j + i * out.weights.col] = (double)((double)rand() / (RAND_MAX / 2)) - 1;
        }
    }

    // init the size of vector
    out.biases = init_vector(length);

    // allocating random doubles to bias
#pragma omp parallel for
    for (int i = 0; i < out.biases.len; i++)
    {
        // random biases from [0, 1)
        out.biases.arr[i] = (double)((double)rand() / (RAND_MAX / 2)) - 1;
    }

    out.activated_outputs = init_vector(length);
    out.weighted_outputs = init_vector(length);

    out.error = init_vector(length);

    return out;
}

void feed_forward(layer_t *current_layer, layer_t *previous_layer)
{
    vector_t weight_inputs = init_vector(current_layer->weights.row);

    // example for second layer, [16x10][10x1]+[16x1]
    multiply_mat_vec(&weight_inputs, &current_layer->weights, &previous_layer->activated_outputs);

    add_vec(&current_layer->weighted_outputs, &weight_inputs, &current_layer->biases);

    free_vector(&weight_inputs);
}

void forward_pass(neural_net_t *net, vector_t *input)
{
    for (int i = 0; i < net->layers[0].length; i++)
    {
        net->layers[0].activated_outputs.arr[i] = input->arr[i];
    }
    for (int i = 1; i < net->num_layers; i++)
    {
        feed_forward(&net->layers[i], &net->layers[i - 1]);
        sigmoid_vec(&net->layers[i].activated_outputs, &net->layers[i].weighted_outputs);
    }
}

void backward_pass(neural_net_t *net, vector_t *expected_outputs)
{
    for (int i = net->num_layers - 1; i > 0; i--)
    {
        if (i == net->num_layers - 1)
        {
            vector_t cost_derivative = init_vector(net->layers[i].length);
            subtract_vec(&cost_derivative, &net->layers[i].activated_outputs, expected_outputs);
            vector_t sigmoid_derivative = init_vector(net->layers[i].length);
            dsigmoid_vec(&sigmoid_derivative, &net->layers[i].weighted_outputs);

            hadamard_product(&net->layers[i].error, &cost_derivative, &sigmoid_derivative);

            free_vector(&cost_derivative);
            free_vector(&sigmoid_derivative);
        }
        else
        {
            matrix_t weights_transpose = init_matrix(net->layers[i + 1].weights.col, net->layers[i + 1].weights.row);
            transpose(&weights_transpose, &net->layers[i + 1].weights);
            vector_t error_product = init_vector(weights_transpose.row);
            multiply_mat_vec(&error_product, &weights_transpose, &net->layers[i + 1].error);

            vector_t sigmoid_derivative = init_vector(net->layers[i].length);
            dsigmoid_vec(&sigmoid_derivative, &net->layers[i].weighted_outputs);

            hadamard_product(&net->layers[i].error, &error_product, &sigmoid_derivative);

            free_matrix(&weights_transpose);
            free_vector(&error_product);
            free_vector(&sigmoid_derivative);
        }
    }
}

void train(neural_net_t *net,
           dataset_t *training_data,
           dataset_t *test_data,
           int epochs, int batch_size,
           double learning_rate)
{
    // Initialize mini-batches
    int num_batches = training_data->num_examples / batch_size;
    dataset_t mini_batches[num_batches];
#pragma omp parallel for
    for (int batch = 0; batch < num_batches; batch++)
    {
        mini_batches[batch].num_examples = batch_size;
        mini_batches[batch].examples = malloc(sizeof(example_t) * batch_size);
    }

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        printf("Epoch %d\n", epoch);
        // shuffle training data
        shuffle_examples(training_data);

        // create mini-batches
        create_mini_batches(training_data, mini_batches, batch_size);

        // Train on mini-batches
        for (int i = 0; i < num_batches; i++)
        {
            train_batch(net, &mini_batches[i], learning_rate);
        }

        // Print progress
        if (epoch % 10 == 0)
        {
            test(net, test_data);
        }
    }

    // Test the network
    test(net, test_data);

#pragma omp parallel for
    for (int batch = 0; batch < num_batches; batch++)
    {
        free(mini_batches[batch].examples);
    }
}

void train_batch(neural_net_t *net, dataset_t *mini_batch, double learning_rate)
{
    matrix_t delta_weights[net->num_layers - 1];
    vector_t delta_biases[net->num_layers - 1];
#pragma omp parallel for
    for (int i = 0; i < net->num_layers - 1; i++)
    {
        delta_weights[i] = init_matrix(net->layers[i + 1].weights.row, net->layers[i + 1].weights.col);
        delta_biases[i] = init_vector(net->layers[i + 1].biases.len);
    }

    for (int i = 0; i < mini_batch->num_examples; i++)
    {
        forward_pass(net, &mini_batch->examples[i].input);
        backward_pass(net, &mini_batch->examples[i].output);

        update_delta_weights(delta_weights, net);
        update_delta_biases(delta_biases, net);
    }
    update_weights(net, delta_weights, learning_rate, mini_batch->num_examples);
    update_biases(net, delta_biases, learning_rate, mini_batch->num_examples);

#pragma omp parallel for
    for (int i = 0; i < net->num_layers - 1; i++)
    {
        free_matrix(&delta_weights[i]);
        free_vector(&delta_biases[i]);
    }
}

void update_delta_weights(matrix_t *weights, neural_net_t *net)
{
    for (int i = net->num_layers - 1; i > 0; i--)
    {
        matrix_t grad = init_matrix(net->layers[i].weights.row, net->layers[i].weights.col);
        multiply_vec_vec(&grad, &net->layers[i].error, &net->layers[i - 1].activated_outputs);
        add_mat(&weights[i - 1], &weights[i - 1], &grad);
        free_matrix(&grad);
    }
}
void update_delta_biases(vector_t *delta_biases, neural_net_t *net)
{
    for (int i = net->num_layers - 1; i > 0; i--)
    {
        add_vec(&delta_biases[i - 1], &delta_biases[i - 1], &net->layers[i].error);
    }
}

void update_weights(neural_net_t *net, matrix_t *delta_weights, double learning_rate, int batch_size)
{
    for (int i = 1; i < net->num_layers; i++)
    {
        scalar_multiply_mat(&delta_weights[i - 1], &delta_weights[i - 1], learning_rate / (double)batch_size);
        subtract_mat(&net->layers[i].weights, &net->layers[i].weights, &delta_weights[i - 1]);
    }
}
void update_biases(neural_net_t *net, vector_t *delta_biases, double learning_rate, int batch_size)
{
    for (int i = 1; i < net->num_layers; i++)
    {
        scalar_multiply_vec(&delta_biases[i - 1], &delta_biases[i - 1], learning_rate / (double)batch_size);
        subtract_vec(&net->layers[i].biases, &net->layers[i].biases, &delta_biases[i - 1]);
    }
}

void test(neural_net_t *net, dataset_t *test_data)
{
    matrix_t network_output = init_matrix(test_data->num_examples, test_data->examples[0].output.len);
    int sum = 0;
    for (int i = 0; i < test_data->num_examples; i++)
    {
        forward_pass(net, &test_data->examples[i].input);
        int max_index = 0;
        for (int j = 0; j < net->layers[net->num_layers - 1].activated_outputs.len; j++)
        {
            if (net->layers[net->num_layers - 1].activated_outputs.arr[j] > net->layers[net->num_layers - 1].activated_outputs.arr[max_index])
            {
                max_index = j;
            }
        }
        if (test_data->examples[i].output.arr[max_index] == 1)
        {
            sum++;
        }
    }
    printf("Accuracy: %d / 10000\n", sum);

    free_matrix(&network_output);
}

void print_matrix(matrix_t *mat)
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
void print_vector(vector_t *vec)
{
    for (int i = 0; i < vec->len; i++)
    {
        printf("%.2lf, ", vec->arr[i]);
    }
    printf("\n");
}

void shuffle_examples(dataset_t *dataset)
{
    for (int i = 0; i < dataset->num_examples; i++)
    {
        int random_index = rand() % dataset->num_examples;
        example_t temp = dataset->examples[i];
        dataset->examples[i] = dataset->examples[random_index];
        dataset->examples[random_index] = temp;
    }
}

void create_mini_batches(dataset_t *dataset, dataset_t *mini_batches, int batch_size)
{
    int num_batches = ceil(dataset->num_examples / batch_size);
#pragma omp parallel for
    for (int batch = 0; batch < num_batches; batch++)
    {
#pragma omp parallel for
        for (int j = 0; j < batch_size; j++)
        {
            if (batch * batch_size + j < dataset->num_examples)
                mini_batches[batch].examples[j] = dataset->examples[batch * batch_size + j];
        }
    }
}
