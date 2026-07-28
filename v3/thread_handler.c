#include "thread_handler.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define IMAGE_SIZE 784
state* initialise_state_arr(int threads, mlp* network,unsigned char* images, unsigned char* labels,int total_img,int img,int* indices)
{
    state* temp=calloc(threads, sizeof(state));
    if(!temp) return NULL;
    for(int i=0;i<threads;i++)
    {
        temp[i].owner=network;
        temp[i].activation=malloc(sizeof(double)*(network->total_biases+IMAGE_SIZE));
        temp[i].dL_dz=malloc(sizeof(double)*(network->total_biases+IMAGE_SIZE)); //first 784 elements of this arr are redundant but we add to maintain similarity of array access
        temp[i].dL_db=malloc(sizeof(double)*(network->total_biases+IMAGE_SIZE)); //first 784 elements of this arr are redundant but we add to maintain similarity of array access
        temp[i].z=malloc(sizeof(double)*(network->total_biases+IMAGE_SIZE)); //first 784 elements of this arr are redundant but we add to maintain similarity of array access
        temp[i].dL_dw=malloc(sizeof(double)*network->total_weights);
        
        if(!temp[i].activation || !temp[i].dL_dz || !temp[i].dL_db || !temp[i].z || !temp[i].dL_dw)
        {
            clear_state_arr(temp, threads);
            return NULL;
        }
        
        temp[i].total_images=total_img;
        temp[i].img_per_state=img;
        temp[i].images=images;
        temp[i].labels=labels;
        temp[i].random_indices=indices;
    }
    return temp;
}
void clear_state_arr(state* arr, int num)
{
    if(!arr) return;
    for(int i=0;i<num;i++)
    {
        free(arr[i].activation);
        free(arr[i].dL_dz);
        free(arr[i].dL_db);
        free(arr[i].dL_dw);
        free(arr[i].z);
    }
    free(arr);
}
void load_image(double* activation,int idx, unsigned char* image)
{
    for(int node=0;node<IMAGE_SIZE;node++)
    {
        activation[node]=image[idx*IMAGE_SIZE+node]/255.0;
    }
}
double sigmoid_func(double x)
{
    return 1.0/(1.0+exp(-x));
}
void* feedforward_and_backprop(void* temp)
{
    state* arg=(state*)temp;
    mlp* net=arg->owner; // local reference to prevent pointer-chasing performance overhead
    int* p_sums=net->p_sums;
    int* w_indices=net->w_indices;
    int* indices=arg->random_indices;

    // Reset thread-local batch gradient buffers
    memset(arg->dL_db,0,sizeof(double)*(net->total_biases+IMAGE_SIZE));
    memset(arg->dL_dw,0,sizeof(double)*net->total_weights);

    // Process worker's partition of the mini-batch
    for(int i=0;i<arg->img_per_state;i++)
    {
        memset(arg->dL_dz,0,sizeof(double)*(net->total_biases+IMAGE_SIZE));
        
        // Randomly sample an image from the dataset partition
        int img_idx=indices[arg->start+i];
        load_image(arg->activation,img_idx,arg->images);
        
        // ------------------ FEEDFORWARD PASS ------------------
        double softmax_temp=0;
        for(int layer=1;layer<net->size;layer++)
        {
            for(int node=0;node<net->summary[layer];node++)
            {
                double temp=0;
                for(int prev_node=0;prev_node<net->summary[layer-1];prev_node++)
                {
                    // Weight matrix mapping between layers: Row = prev_node, Col = node. Stride = size(layer)
                    temp+=net->weights[w_indices[layer-1]+prev_node*net->summary[layer]+node]*arg->activation[p_sums[layer-1]+prev_node];
                }
                arg->z[p_sums[layer]+node]=temp+net->biases[net->p_sums[layer]+node];
                
                if(layer!=net->size-1)
                {
                    // Apply Sigmoid activation function to hidden layers
                    arg->activation[p_sums[layer]+node]=sigmoid_func(arg->z[p_sums[layer]+node]);
                }
                else 
                {
                    // Sum exponential outputs for Softmax activation at the output layer
                    softmax_temp+=exp(arg->z[p_sums[layer]+node]);
                }
            }
        }
        
        // Finalize Softmax probability output
        int last_layer=net->size-1;
        for(int node=0;node<net->summary[last_layer];node++) 
        {
            arg->activation[p_sums[last_layer]+node]=exp(arg->z[p_sums[last_layer]+node])/softmax_temp;
        } 

        // ------------------ BACKPROPAGATION PASS ------------------
        // Loop backwards starting from output layer down to the first hidden layer
        for(int layer=last_layer;layer>0;layer--) 
        {
            for(int node=0;node<net->summary[layer];node++)
            {
                if(layer==last_layer)
                {
                    // For Softmax + Cross-Entropy, output derivative simplifies to: dL/dz = activation - target
                    int y = (arg->labels[img_idx] == node) ? 1 : 0;
                    arg->dL_dz[p_sums[layer]+node]=arg->activation[p_sums[layer]+node]-y;
                }
                else
                {
                    // Hidden layer error delta propagation: delta = sum(delta_next * weight_next) * sigmoid'(z)
                    // Row = node (current layer), Col = next_node (next layer). Stride = size(layer+1)
                    for(int next_node=0;next_node<net->summary[layer+1];next_node++)
                    {
                        arg->dL_dz[p_sums[layer]+node]+=arg->dL_dz[p_sums[layer+1]+next_node]*net->weights[w_indices[layer]+node*net->summary[layer+1]+next_node]*arg->activation[p_sums[layer]+node]*(1-arg->activation[p_sums[layer]+node]);
                    }
                }
                // Accumulate local bias gradients: dL/db = delta
                arg->dL_db[p_sums[layer]+node]+=arg->dL_dz[p_sums[layer]+node];
            }
        }
        
        // Accumulate local weight gradients: dL/dw = delta_next * activation_current
        for(int layer=last_layer-1;layer>=0;layer--)
        {
            for(int node=0;node<net->summary[layer];node++)
            {
                for(int next_node=0;next_node<net->summary[layer+1];next_node++)
                {
                    arg->dL_dw[w_indices[layer]+node*net->summary[layer+1]+next_node]+=arg->dL_dz[p_sums[layer+1]+next_node]*arg->activation[p_sums[layer]+node];
                }
            }
        }

    }
    return NULL;
}





