#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifndef NN_MATH_H
#define NN_MATH_H

/**
 * @brief A matrix struct
 */
typedef struct
{
    double *arr; /// The array of the matrix
    int row;     /// The number of rows in the matrix
    int col;     /// The number of columns in the matrix
} matrix_t;

/**
 * @brief A vector struct
 */
typedef struct
{
    double *arr; /// The array of the vector
    int len;     /// The length of the vector
} vector_t;

/**
 * @brief Fincion to create and allocate a vector
 *
 * @param len The length of the vector
 *
 * @return vector_t The vector
 */
vector_t init_vector(int len);

/**
 * @brief Function to create and allocate a matrix
 *
 * @param row The number of rows in the matrix
 * @param col The number of columns in the matrix
 *
 * @return matrix_t The matrix
 */
matrix_t init_matrix(int row, int col);

/**
 * @brief Function to allocate the memory for a vector
 *
 * @param len The length of the vector
 *
 * @return double* The vector
 */
double *allocate_vec_arr(int len);

/**
 * @brief Function to allocate the memory for a matrix
 *
 * @param rows The number of rows in the matrix
 * @param cols The number of columns in the matrix
 *
 * @return double* The matrix
 */
double *allocate_mat_arr(int rows, int cols);

/**
 * @brief Function to free the memory of a vector
 *
 * @param vec The vector to free
 */
void free_vector(vector_t *vec);

/**
 * @brief Function to free the memory of a matrix
 *
 * @param mat The matrix to free
 */
void free_matrix(matrix_t *mat);

/**
 * @brief Function to multiply a matrix and a vector
 *
 * @param out The output vector
 * @param mat The matrix
 * @param vec The vector
 */
void multiply_mat_vec(vector_t *out, matrix_t *mat, vector_t *vec);

/**
 * @brief Function to add two vectors
 *
 * @param out The output vector
 * @param v1 The first vector
 * @param v2 The second vector
 */
void add_vec(vector_t *out, vector_t *v1, vector_t *v2);

/**
 * @brief Function to subtract two vectors
 *
 * @param out The output vector
 * @param v1 The first vector
 * @param v2 The second vector
 */
void subtract_vec(vector_t *out, vector_t *v1, vector_t *v2);

/**
 * @brief Function to get the sigmoid of a value
 *
 * @param val The value to get the sigmoid of
 *
 * @return double The sigmoid of the value
 */
double sigmoid(double val);

/**
 * @brief Function to get the ReLU of a value
 *
 * @param val The value to get the ReLU of
 *
 * @return double The ReLU of the value
 */
double ReLU(double val);

/**
 * @brief Function to get the sigmoid of a matrix
 *
 * @param out The output matrix
 * @param mat The input matrix
 */
void sigmoid_mat(matrix_t *out, matrix_t *mat);

/**
 * @brief Function to get the sigmoid derivative of a matrix
 *
 * @param out The output matrix
 * @param mat The input matrix
 */
void dsigmoid(matrix_t *out, matrix_t *mat);

/**
 * @brief Function to get the ReLU of a matrix
 *
 * @param out The output matrix
 * @param mat The input matrix
 */
void ReLU_mat(matrix_t *out, matrix_t *mat);

/**
 * @brief Function to get the ReLU derivative of a matrix
 *
 * @param out The output matrix
 * @param mat The input matrix
 */
void dReLU(matrix_t *out, matrix_t *mat);

/**
 * @brief Function to get the sigmoid of a vector
 *
 * @param out The output vector
 * @param vec The input vector
 */
void sigmoid_vec(vector_t *out, vector_t *vec);

/**
 * @brief Function to get the sigmoid derivative of a vector
 *
 * @param out The output vector
 * @param vec The input vector
 */
void dsigmoid_vec(vector_t *out, vector_t *vec);

/**
 * @brief Function to get the ReLU of a vector
 *
 * @param out The output vector
 * @param vec The input vector
 */
void ReLU_vec(vector_t *out, vector_t *vec);

/**
 * @brief Function to get the ReLU derivative of a vector
 *
 * @param out The output vector
 * @param vec The input vector
 */
void dReLU_vec(vector_t *out, vector_t *vec);

/**
 * @brief Function to get the hadamard product of two vectors
 *
 * @param out The output vector
 * @param vec1 The first vector
 * @param vec2 The second vector
 */
void hadamard_product(vector_t *out, vector_t *vec1, vector_t *vec2);

/**
 * @brief Function to get the transpose of a matrix
 *
 * @param out The output matrix
 * @param mat The input matrix
 */
void transpose(matrix_t *out, matrix_t *mat);

/**
 * @brief Function to multiply a row vector and a column vector
 *
 * @param out The output matrix
 * @param v1 The row vector
 * @param v2 The column vector
 */
void multiply_vec_vec(matrix_t *out, vector_t *v1, vector_t *v2);

/**
 * @brief Function to multiply a matrix and a scalar
 *
 * @param out The output matrix
 * @param mat The input matrix
 * @param scalar The scalar
 */
void scalar_multiply_mat(matrix_t *out, matrix_t *mat, double scalar);

/**
 * @brief Function to subtract two matrices
 *
 * @param out The output matrix
 * @param mat1 The first matrix
 * @param mat2 The second matrix
 */
void subtract_mat(matrix_t *out, matrix_t *mat1, matrix_t *mat2);

/**
 * @brief Function to multiply a vector and a scalar
 *
 * @param out The output vector
 * @param vec The input vector
 * @param scalar The scalar
 */
void scalar_multiply_vec(vector_t *out, vector_t *vec, double scalar);

/**
 * @brief Function to add two matrices
 *
 * @param out The output matrix
 * @param mat1 The first matrix
 * @param mat2 The second matrix
 */
void add_mat(matrix_t *out, matrix_t *mat1, matrix_t *mat2);

#endif
