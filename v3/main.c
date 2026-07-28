#include <stdio.h>
#include "file_handler.h"
#include "mlp.h"
#include "thread_handler.h"
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#define IMAGE_SIZE 784
#define MAX_PIXEL_VAL 255
#if defined(_WIN32)
    #include <windows.h>

#else
    #include <unistd.h>
#endif
int get_cores()
{
    #if defined(_WIN32)
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        return (int)sysinfo.dwNumberOfProcessors;
    #else
        long count=sysconf(_SC_NPROCESSORS_ONLN);

        return count>0?(int)count:1;
    #endif
}
void shuffle_images_and_labels(int num,int* index)
{
    int temp;
    for(int i=num-1;i>0;i--)
    {
        int swap=abs(rand())%(i+1);
        temp=index[i];
        index[i]=index[swap];
        index[swap]=temp;
    }
}
int main()
{
    srand(time(NULL));
    image_file train_images_file=read_image_file("mnist/train-images-idx3-ubyte"); 
    if(train_images_file.file==NULL)
    {
        return 1;
    }
    label_file train_labels_file=read_label_file("mnist/train-labels-idx1-ubyte");
    unsigned char* train_images_pixels=malloc(sizeof(unsigned char)*train_images_file.images_num*IMAGE_SIZE); //training image pixel values
    if(fread(train_images_pixels,sizeof(unsigned char),IMAGE_SIZE*train_images_file.images_num,train_images_file.file)!=IMAGE_SIZE*train_images_file.images_num)
    {
        printf("Read unsuccessfull\n");
        return 1;
    }
    unsigned char* train_labels=malloc(sizeof(unsigned char)*train_labels_file.labels_num); //training set labels
    if(fread(train_labels,sizeof(unsigned char),train_labels_file.labels_num,train_labels_file.file)!=train_labels_file.labels_num)
    {
        printf("Read unsuccessfull\n");
        return 1;
    }
    image_file test_images_file=read_image_file("mnist/t10k-images-idx3-ubyte");
    if(test_images_file.file==NULL)
    {
        return 1;
    }
    unsigned char* test_images_pixels=malloc(sizeof(unsigned char)*test_images_file.images_num*IMAGE_SIZE); //test set pixel values
    if(fread(test_images_pixels, sizeof(unsigned char), IMAGE_SIZE*test_images_file.images_num, test_images_file.file)!=IMAGE_SIZE*test_images_file.images_num)
    {
        printf("Read unsuccessfull\n");
        return 1;
    }
    label_file test_labels_file=read_label_file("mnist/t10k-labels-idx1-ubyte");
    unsigned char* test_labels=malloc(sizeof(unsigned char)*test_labels_file.labels_num);
    if(fread(test_labels,sizeof(unsigned char),test_labels_file.labels_num,test_labels_file.file)!=test_labels_file.labels_num)
    {
        printf("Read unsuccessfull\n");
        return 1;
    }
    fclose(test_images_file.file);
    fclose(train_images_file.file);
    fclose(train_labels_file.file);
    fclose(test_labels_file.file);

    int* indices=malloc(sizeof(int)*train_labels_file.labels_num);
    if(!indices)
    {
        fprintf(stderr,"Error: Failed to allocate index array.\n");
        return 1;
    }
    for(unsigned int i=0;i<train_labels_file.labels_num;i++)
    {
        indices[i]=i;
    }

    printf("Enter the number of hidden layers you want in your MLP for MNIST dataset: ");
    int num_layers; // input and output layers
    scanf("%d",&num_layers);
    num_layers+=2;
    int mlp_info[num_layers];
    mlp_info[0]=IMAGE_SIZE;
    mlp_info[num_layers-1]=10;
    for(int i=1;i<num_layers-1;i++)
    {
        printf("Enter number of nodes in hidden layer %d: ",i-1);
        scanf("%d",&mlp_info[i]);
    }
    mlp* network=create_mlp(mlp_info, num_layers);
    if(!network)
    {
        fprintf(stderr,"Error: Failed to allocate MLP network memory.\n");
        return 1;
    }
    initialise_network(network); //set weights and activations to random, maintaining xavier-glorot range

    double learning_rate;
    int batch_size, epochs;
    printf("Enter desired learning rate: ");
    scanf("%lf",&learning_rate);
    printf("Enter desired bacth size (preferred like 2,4,16,32...): ");
    scanf("%d",&batch_size);
    printf("Enter desired epochs: ");
    scanf("%d",&epochs);
    int avl_cores=get_cores();
    int num_threads_ideal=(avl_cores>batch_size?batch_size:avl_cores); //number of threads is preferred to be equal to avl cores
    printf("avl cores: %d\n",avl_cores);

    state* buffer_arr=initialise_state_arr(num_threads_ideal, network,train_images_pixels,train_labels,train_images_file.images_num,0,indices); //number of state buffers is equal to threads, each buffer will be passed to a thread
    pthread_t t_arr[num_threads_ideal]; //thread array

    printf("Starting training...\n");
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for(int e=0;e<epochs;e++)
    {
        shuffle_images_and_labels(train_images_file.images_num,indices);
        int batches=train_images_file.images_num/batch_size;
        for(int b=0;b<batches;b++)
        {
            // Allocate mini-batch partitions dynamically across active worker threads
            for(int thread=0;thread<num_threads_ideal;thread++)
            {
                buffer_arr[thread].start=b*batch_size+(batch_size/num_threads_ideal)*thread;
                if(thread==num_threads_ideal-1)
                {
                    // Give remaining leftover images to the last thread to balance work
                    buffer_arr[thread].img_per_state=batch_size/num_threads_ideal+batch_size%num_threads_ideal; 
                }
                else
                {
                    buffer_arr[thread].img_per_state=batch_size/num_threads_ideal;
                }
                // Spawn worker thread to run feedforward + backprop on its partition
                if(pthread_create(&t_arr[thread],NULL,feedforward_and_backprop,&buffer_arr[thread])!=0)
                {
                    fprintf(stderr,"Error: Failed to create thread %d\n",thread);
                    return 1;
                }
            }
            
            // Barrier synchronization: Wait for all thread workers to finish their passes
            for(int thread=0;thread<num_threads_ideal;thread++)
            {
                pthread_join(t_arr[thread],NULL);
            }
            
            // Gradient aggregation: Sum local thread bias gradients and perform SGD update
            for(int i=IMAGE_SIZE;i<network->total_biases;i++)
            {
                double temp=0;
                for(int t=0;t<num_threads_ideal;t++)
                {
                    temp+=buffer_arr[t].dL_db[i];
                }
                network->biases[i]-=(learning_rate*temp)/batch_size;
            }
            
            // Gradient aggregation: Sum local thread weight gradients and perform SGD update
            for(int i=0;i<network->total_weights;i++)
            {
                double temp=0;
                for(int t=0;t<num_threads_ideal;t++)
                {
                    temp+=buffer_arr[t].dL_dw[i];
                }
                network->weights[i]-=(learning_rate*temp)/batch_size;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_time=(end_time.tv_sec-start_time.tv_sec)+(end_time.tv_nsec-start_time.tv_nsec)/1e9;
    printf("Training Time (New Parallelized): %f seconds\n",elapsed_time);
    printf("Training finished\n");
    printf("Evaluating on test set...\n");
    state* eval=initialise_state_arr(1, network, test_images_pixels, test_labels, test_images_file.images_num,1,NULL);
    mlp* eval_net=eval->owner;
    int* eval_p_sums=eval_net->p_sums;
    int* eval_w_indices=eval_net->w_indices;
    int eval_last_layer=eval_net->size-1;
    double accuracy=0;
    for(unsigned int image=0;image<test_images_file.images_num;image++)
    {
        for(int layer=0;layer<eval_net->size;layer++)
        {
            if(layer==0)
            {
                load_image(eval->activation, image, test_images_pixels);
            }
            else
            {
                for(int node=0;node<eval_net->summary[layer];node++)
                {
                    double temp=0;
                    for(int prev_node=0;prev_node<eval_net->summary[layer-1];prev_node++)
                    {
                        temp+=eval->activation[eval_p_sums[layer-1]+prev_node]*eval_net->weights[eval_w_indices[layer-1]+prev_node*eval_net->summary[layer]+node];
                    }
                    temp+=eval_net->biases[eval_p_sums[layer]+node];
                    eval->activation[eval_p_sums[layer]+node]=sigmoid_func(temp);
                }
            }
        }
        int prediction=0;
        for(int i=0;i<eval_net->summary[eval_last_layer];i++)
        {
            if(eval->activation[eval_p_sums[eval_last_layer]+i]>eval->activation[eval_p_sums[eval_last_layer]+prediction])
            {
                prediction=i;
            }
        }
        if(prediction==test_labels[image])
        {
            accuracy++;
        }
    }
    accuracy/=test_images_file.images_num;
    printf("%f accuracy on %d epochs, %d batch size, %f learning rate\n",accuracy*100,epochs,batch_size,learning_rate);
    free(indices);
    free(train_images_pixels);
    free(train_labels);
    free(test_images_pixels);
    free(test_labels);
    clear_network(network);
    clear_state_arr(buffer_arr, num_threads_ideal);
    clear_state_arr(eval, 1);
    return 0;
    
}