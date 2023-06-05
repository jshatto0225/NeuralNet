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

    //initalizing them
    init_layer(&input, INPUT_SIZE,HIDDEN_SIZE, &weights1, &bias1);
    init_layer(&hidden1, HIDDEN_SIZE,HIDDEN_SIZE, &weights2, &bias2);
    init_layer(&hidden2, HIDDEN_SIZE,OUTPUT_SIZE, &weights3, &bias3);
    init_layer(&output, OUTPUT_SIZE,0, &weights4, &bias4);


    return 0;
}