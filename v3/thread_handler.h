#include "mlp.h"
#pragma once
typedef struct state
{
    mlp* owner;
    unsigned char* images;
    unsigned char* labels;
    int total_images;
    int img_per_state;
    double* activation;
    double* z;
    double* dL_dz;
    double* dL_db;
    double* dL_dw;
    int start;
    int* random_indices;
}state;
state* initialise_state_arr(int threads, mlp* network,unsigned char* images, unsigned char* labels,int total_img,int img,int* indices);
void clear_state_arr(state* arr, int num);
void* feedforward_and_backprop(void* temp);
void load_image(double* activation,int idx, unsigned char* image);
double sigmoid_func(double x);
