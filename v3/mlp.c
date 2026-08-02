#include <stdlib.h>
#include "mlp.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#define IMAGE_SIZE 784
mlp* create_mlp(int* arr, int layers)
{
    mlp* temp=malloc(sizeof(mlp));
    if(!temp)
    {
        return NULL;
    }

    temp->size=layers;
    temp->summary=malloc(sizeof(int)*layers); // Stores the number of nodes per layer
    temp->p_sums=malloc(sizeof(int)*layers); // Stores bias/activation offset prefix sums for flat 1D allocation
    temp->w_indices=malloc(sizeof(int)*(layers-1)); // Stores weight offsets mapping for transitions between layers
    if(!temp->summary || !temp->p_sums || !temp->w_indices)
    {
        clear_network(temp);
        return NULL;
    }
    for(int layer=0;layer<layers;layer++)
    {
        temp->summary[layer]=arr[layer];
        if(layer==0)
        {
            temp->p_sums[layer]=0;
            temp->w_indices[layer]=0;
        }
        else
        {
            if(layer!=layers-1)
            {
                // Weight offsets: cumulative sum of weights in previous layer transitions (nodes_current * nodes_previous)
                temp->w_indices[layer]=temp->w_indices[layer-1]+arr[layer]*arr[layer-1];
            }
            // Bias/activation offsets: prefix sum to map index to start of each layer's block
            temp->p_sums[layer]=temp->p_sums[layer-1]+arr[layer-1]; // indices of z,a,b,etc: arr[p_sums[layer]+node]
        }
    }
    // Calculate total allocation sizes for flat contiguous arrays
    temp->total_biases=temp->p_sums[layers-1]+arr[layers-1];
    temp->total_weights=temp->w_indices[layers-2]+arr[layers-1]*arr[layers-2];
    
    // Contiguous bias allocation (padded by IMAGE_SIZE to keep input indexing aligned)
    temp->biases=malloc(sizeof(double)*(temp->total_biases+IMAGE_SIZE)); 
    temp->weights=malloc(sizeof(double)*temp->total_weights);
    if(!temp->biases || !temp->weights)
    {
        clear_network(temp);
        return NULL;
    }
    return temp;
}
void initialise_network(mlp* temp)
{
    int* weights=temp->w_indices;
    double* w=temp->weights;
    int* info=temp->summary;
    for(int layer=0;layer<temp->size-1;layer++)
    {
        double range=sqrt(6.0/(info[layer]+info[layer+1]));
        for(int node=0;node<info[layer];node++)
        {
            for(int next_node=0;next_node<info[layer+1];next_node++)
            {
                int index=weights[layer]+node*info[layer+1]+next_node;
                double tempp=(double)rand()/RAND_MAX;
                tempp=tempp*2-1;
                w[index]=tempp*range;
            }
        }
    }
    memset(temp->biases,0,temp->total_biases*sizeof(w[0]));
}
void clear_network(mlp* temp)
{
    free(temp->biases);
    free(temp->weights);
    free(temp->p_sums);
    free(temp->summary);
    free(temp->w_indices);
    free(temp);
    return;
}
int save_network(mlp* net, const char* path)
{
    FILE* f=fopen(path,"wb");
    if(!f)
    {
        return 0;
    }
    fwrite(&net->size,sizeof(int),1,f);
    fwrite(&net->total_weights,sizeof(int),1,f);
    fwrite(&net->total_biases,sizeof(int),1,f);
    fwrite(net->summary,sizeof(int),net->size,f);
    fwrite(net->weights,sizeof(double),net->total_weights,f);
    fwrite(net->biases,sizeof(double),net->total_biases,f);
    fclose(f);
    return 1;
}