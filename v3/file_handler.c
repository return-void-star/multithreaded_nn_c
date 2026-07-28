#include "file_handler.h"
#include <stdlib.h>
unsigned int read_big_endian_uint(FILE* fp)
{
    unsigned char bytes[4];
    fread(bytes, sizeof(unsigned char), 4, fp);
    return (bytes[0]<<24)|(bytes[1]<<16)|(bytes[2]<<8)|bytes[3];
}
image_file read_image_file(char* temp)
{
    image_file x;
    x.file=fopen(temp,"rb");
    if(x.file==NULL)
    {
        printf("Image file read failed\n");
        return x;
    }
    x.magic_num=read_big_endian_uint(x.file);
    x.images_num=read_big_endian_uint(x.file);
    x.rows=read_big_endian_uint(x.file);
    x.cols=read_big_endian_uint(x.file);
    return x;
}
label_file read_label_file(char* temp)
{
    label_file x;
    x.file=fopen(temp,"rb");
    if(x.file==NULL)
    {
        printf("Label file read failed\n");
        return x;
    }
    x.magic_num=read_big_endian_uint(x.file);
    x.labels_num=read_big_endian_uint(x.file);
    return x;
}