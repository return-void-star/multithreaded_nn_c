#pragma once
typedef struct mlp
{
    int size;
    int* summary;
    int* p_sums;
    int* w_indices;
    double* biases;
    double* weights;
    int total_biases;
    int total_weights;
}mlp;
mlp* create_mlp(int* arr, int layers);
void initialise_network(mlp* temp);
void clear_network(mlp* temp);