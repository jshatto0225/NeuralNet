#include "NeuralNet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dataset.h"
#include "mnist.h"
#include <omp.h>

/**
 * @brief Function to initialize the training data from the MNIST dataset
 *
 * @return dataset_t The training data
 */
dataset_t init_train_data()
{
    dataset_t mnist_data;
    mnist_data.num_examples = NUM_TRAIN;
    mnist_data.examples = malloc(sizeof(example_t) * mnist_data.num_examples);

    for (int i = 0; i < NUM_TRAIN; i++)
    {
        mnist_data.examples[i].input = init_vector(SIZE);
        mnist_data.examples[i].output = init_vector(10);
        for (int j = 0; j < SIZE; j++)
        {
            mnist_data.examples[i].input.arr[j] = train_image[i][j];
        }
        mnist_data.examples[i].output.arr[train_label[i]] = 1;
    }
    return mnist_data;
}

/**
 * @brief Function to initialize the test data from the MNIST dataset
 *
 * @return dataset_t The test data
 */
dataset_t init_test_data()
{
    dataset_t mnist_data;
    mnist_data.num_examples = NUM_TEST;
    mnist_data.examples = malloc(sizeof(example_t) * mnist_data.num_examples);

    for (int i = 0; i < NUM_TEST; i++)
    {
        mnist_data.examples[i].input = init_vector(SIZE);
        mnist_data.examples[i].output = init_vector(10);
        for (int j = 0; j < SIZE; j++)
        {
            mnist_data.examples[i].input.arr[j] = test_image[i][j];
        }
        mnist_data.examples[i].output.arr[test_label[i]] = 1;
    }
    return mnist_data;
}

/**
 * @brief Function to free a dataset
 *
 * @param dataset The dataset to free
 */
void free_dataset(dataset_t *dataset)
{
    for (int i = 0; i < dataset->num_examples; i++)
    {
        free_vector(&dataset->examples[i].input);
        free_vector(&dataset->examples[i].output);
    }
    free(dataset->examples);
}

int main()
{
    // Seed random number generator
    srand(time(NULL));

    // Network architecture
    int sizes[] = {784, 30, 10};

    // Load MNIST dataset
    load_mnist();

    // Allocate network
    neural_net_t net = allocate_neural_net(sizeof(sizes) / sizeof(int), sizes);
    printf("Allocated Network\n");

    // Initialize training and test data
    dataset_t train_data = init_train_data();
    dataset_t test_data = init_test_data();
    printf("Initialized Data\n");

    // Train network
    train(&net, &train_data, &test_data, 30, 10, 3.0);

    // Free data and network
    free_dataset(&train_data);
    free_dataset(&test_data);
    free_network(&net);

    return 0;
}

/*
int main()
{
    srand(time(NULL));

    srand(time(NULL));

    int sizes[] = {784, 64, 64, 10};

    load_mnist();

    neural_net_t net = allocate_neural_net(sizeof(sizes) / sizeof(int), sizes);
    printf("Allocated Network\n");

    // TODO: Shuffle data
    matrix_t x_train = init_matrix(NUM_TRAIN, SIZE);
    matrix_t y_train = init_matrix(NUM_TRAIN, 10);

    matrix_t x_test = init_matrix(NUM_TEST, SIZE);
    matrix_t y_test = init_matrix(NUM_TEST, 10);

    matrix_t test_input = init_matrix(1, SIZE);
    matrix_t test_output = init_matrix(1, 10);

    load_mnist_matrix_vector(&x_train, &y_train, &x_test, &y_test);

    matrix_t test = init_matrix(1, SIZE);
    matrix_t test2 = init_matrix(1, 10);

    for (int i = 0; i < SIZE; i++)
    {
        test.arr[i] = x_train.arr[i];
    }
    for (int i = 0; i < 10; i++)
    {
        test2.arr[i] = y_train.arr[i];
    }

    train(&net, &test, &test2, 1, 1, 0.1, &x_test, &y_test);

    return 0;
}*/