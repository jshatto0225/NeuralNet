#include "NeuralNet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dataset.h"

#define INPUT_SIZE 40
#define HIDDEN_SIZE 10
#define OUTPUT_SIZE 40

int main()
{
    //making each layer, weight, and bias
    struct layer input, hidden1, hidden2, output;
    struct matrix weights1, weights2, weights3, weights4;
    struct vector bias1, bias2, bias3, bias4;
    struct vector nodes1, nodes2, nodes3, nodes4;
    struct vector activ1, activ2, activ3, activ4;

    double *X_train = x_train();
    double *Y_train = y_train(X_train);
    
    //initalizing them
    init_layer(&input, INPUT_SIZE,HIDDEN_SIZE, &weights1, &bias1, &nodes1, &activ1);
    init_layer(&hidden1, HIDDEN_SIZE,HIDDEN_SIZE, &weights2, &bias2, &nodes2, &activ2);
    init_layer(&hidden2, HIDDEN_SIZE,OUTPUT_SIZE, &weights3, &bias3, &nodes3, &activ3);
    init_layer(&output, OUTPUT_SIZE,0, &weights4, &bias4,&nodes4, &activ4);


    // printf("Weights:\n");
    struct matrix matrix = weights1;
    struct vector vector = activ4;



    //"inputting first layer"
        for (int i = 0; i < 40; i++)
        {
            nodes1.arr[i] = X_train[i];
        }


    struct layer* layers = malloc(4*sizeof(struct layer));
    layers[0] = input;
    layers[1] = hidden1;
    layers[2] = hidden2;
    layers[3] = output;
 

    //training loop
    int epochs = 99;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        activation(layers, 4);

    }   

    
    for (int i = 0; i < 10; i++)
    {
        printf("%lf\n", nodes4.arr[i]);
    }


    
    
    // for (int i = 0; i < matrix.row; i++)
    // {
    //     for (int j = 0; j < matrix.col; j++)
    //     {
    //         printf("%.2lf, ", matrix.arr[j + i * matrix.col]);
    //     }
    //     printf("\n");
    // }
    // printf("Weights dim: %d X %d\n", matrix.row, matrix.col);

    // printf("Biases:\n");
    //     for (int i = 0; i < vector.len; i++)
    //     {
    //         printf("%.2lf\n", vector.arr[i]);
    //     }
    // printf("Bias dim: %d\n", vector.len);
    

    //freeing allocated space
    free_layer(&weights1, &bias1, &nodes1, &activ1);
    free_layer(&weights2, &bias2, &nodes2, &activ2);
    free_layer(&weights3, &bias3, &nodes3, &activ3);
    free_layer(&weights4, &bias4, &nodes4, &activ4);
    free(layers);
    free(X_train);
    free(Y_train);


    printf("Your code has no errors!\n");
    return 0;
}