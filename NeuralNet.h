#ifndef NURAL_NET
#define NEURAL_NET

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nnMath.h"
#include <time.h>

/**
 * @brief A layer struct
 */
typedef struct
{
    matrix_t weights;           /// The weights of the layer
    vector_t biases;            /// The biases of the layer
    vector_t weighted_outputs;  /// The weighted outputs of the layer
    vector_t activated_outputs; /// The activated outputs of the layer
    vector_t error;             /// The error of the layer
    int length;                 /// The length of the layer
} layer_t;

/**
 * @brief A neural network struct
 */
typedef struct neuralnet
{
    layer_t *layers; /// The layers of the network
    int num_layers;  /// The number of layers in the network
} neural_net_t;

/**
 * @brief A struct to hold the input and output of an example
 */
typedef struct
{
    vector_t input;  /// The input of the example
    vector_t output; /// The output of the example
} example_t;

/**
 * @brief A struct to hold a dataset
 */
typedef struct
{
    example_t *examples; /// The examples in the dataset
    int num_examples;    /// The number of examples in the dataset
} dataset_t;

/**
 * @brief Function to print a matrix
 *
 * @param mat The matrix to print
 */
void print_matrix(matrix_t *mat);

/**
 * @brief Function to print a vector
 *
 * @param vec The vector to print
 */
void print_vector(vector_t *vec);

/**
 * @brief Function to initialize a neural network
 *
 * @param length The number of layers in the network
 * @param sizes The sizes of each layer
 *
 * @return A neural network
 */
neural_net_t allocate_neural_net(int length, int *sizes);

/**
 * @brief Function to initialize a layer
 *
 * @param length The length of the layer
 * @param previous_layer_length The length of the previous layer
 *
 * @return A layer
 */
layer_t init_layer(int length, int previous_layer_length);

/**
 * @brief Function to free a layer
 *
 * @param layer The layer to free
 */
void free_layer(layer_t *layer);

/**
 * @brief Function to free a neural network
 *
 * @param net The neural network to free
 */
void free_network(neural_net_t *net);

/**
 * @brief Function to get the output of a layer
 *
 * @param layer The layer to get the output of
 * @param previous_layer The previous layer
 */
void feed_forward(layer_t *current_layer, layer_t *previous_layer);

/**
 * @brief Function to get the output of a neural network
 *
 * @param network The neural network to get the output of
 * @param input The input to the neural network
 */
void forward_pass(neural_net_t *network, vector_t *input);

/**
 * @brief Function to get the error of a neural network
 *
 * @param network The neural network to get the error of
 * @param expected_outputs The expected outputs of the neural network
 */
void backward_pass(neural_net_t *network, vector_t *expected_outputs);

/**
 * @brief Function to train a neural network
 *
 * @param net The neural network to train
 * @param training_data The training data
 * @param test_data The test data
 * @param epochs The number of epochs to train for
 * @param batch_size The size of the mini batches
 * @param learning_rate The learning rate
 */
void train(neural_net_t *net,
           dataset_t *training_data,
           dataset_t *test_data,
           int epochs, int batch_size,
           double learning_rate);

/**
 * @brief Function to test the accuracy of a neural network
 *
 * @param net The neural network to test
 * @param test_data The test data
 */
void test(neural_net_t *net, dataset_t *test_data);

/**
 * @brief Function to train a neural network on a batch of data
 *
 * @param net The neural network to train
 * @param batch_training_data The batch of training data
 * @param learning_rate The learning rate
 */
void train_batch(neural_net_t *net, dataset_t *batch_training_data, double learning_rate);

/**
 * @brief Function to shuffle the examples in a dataset
 *
 * @param dataset The dataset to shuffle
 */
void shuffle_examples(dataset_t *dataset);

/**
 * @brief Function to create mini batches from a dataset
 *
 * @param dataset The dataset to create mini batches from
 * @param mini_batches The mini batches to create
 * @param batch_size The size of the mini batches
 */
void create_mini_batches(dataset_t *dataset, dataset_t *mini_batches, int batch_size);

/**
 * @brief Function to calculate the change in weights
 *
 * @param weights The weights to update
 * @param net The neural network
 */
void update_delta_weights(matrix_t *weights, neural_net_t *net);

/**
 * @brief Function to calculate the change in biases
 *
 * @param biases The biases to update
 * @param net The neural network
 */
void update_delta_biases(vector_t *biases, neural_net_t *net);

/**
 * @brief Function to update the weights of a neural network
 *
 * @param net The neural network
 * @param delta_weights The change in weights
 * @param learning_rate The learning rate
 * @param batch_size The size of the mini batches
 */
void update_weights(neural_net_t *net, matrix_t *delta_weights, double learning_rate, int batch_size);

/**
 * @brief Function to update the biases of a neural network
 *
 * @param net The neural network
 * @param delta_biases The change in biases
 * @param learning_rate The learning rate
 * @param batch_size The size of the mini batches
 */
void update_biases(neural_net_t *net, vector_t *delta_biases, double learning_rate, int batch_size);

#endif