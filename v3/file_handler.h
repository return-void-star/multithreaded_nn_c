#include <stdio.h>
#pragma once
typedef struct image_file
{
    FILE* file;
    unsigned int magic_num,images_num,rows,cols;
}image_file;
typedef struct label_file
{
    FILE* file;
    unsigned int magic_num, labels_num;
}label_file;

image_file read_image_file(char* temp);
label_file read_label_file(char* temp);