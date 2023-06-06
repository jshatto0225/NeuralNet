#include "NeuralNet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 10
#define HIDDEN_SIZE 16
#define OUTPUT_SIZE 10

int main()
{
    //making each layer, weight, and bias
    struct layer input, hidden1, hidden2, output;
    struct matrix weights1, weights2, weights3, weights4;
    struct vector bias1, bias2, bias3, bias4;
    struct vector nodes1, nodes2, nodes3, nodes4;

    //"inputting first layer"
    nodes1.arr = (double *)malloc(40 * sizeof(double));
    for (int i = 0; i < 40; i++)
    {
        nodes1.arr[i] = i;
    }



    //initalizing them
    init_layer(&input, INPUT_SIZE,HIDDEN_SIZE, &weights1, &bias1, &nodes1);
    init_layer(&hidden1, HIDDEN_SIZE,HIDDEN_SIZE, &weights2, &bias2, &nodes2);
    init_layer(&hidden2, HIDDEN_SIZE,OUTPUT_SIZE, &weights3, &bias3, &nodes3);
    init_layer(&output, OUTPUT_SIZE,0, &weights4, &bias4,&nodes4);


    struct layer* layers = malloc(4*sizeof(struct layer));
    layers[0] = input;
    layers[1] = hidden1;
    layers[2] = hidden2;
    layers[3] = output;
 

    //training loop
    int epochs = 99;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        //forward propagation
        struct vector y_pred = activation(layers, 4);

        //calculate loss
        struct vector loss;

        

    }   
    

    //freeing allocated space
    free_layer(&weights1, &bias1);
    free_layer(&weights2, &bias2);
    free_layer(&weights3, &bias3);
    free_layer(&weights4, &bias4);
    free((nodes1.arr));
    free(layers);
    printf("Your code has no errors!\n");
    return 0;
}