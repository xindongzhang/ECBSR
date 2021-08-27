#include "tensor.h"

int32_t create_float_tensor(FloatTensor **t, Shape shape, float_t val)
{
    *t = (FloatTensor*) malloc(sizeof(FloatTensor));
    (*t)->shape = shape;
    (*t)->size = shape.n*shape.h*shape.w*shape.c;
    // 8 bit align
    // int32_t align_size = (*t)->size + (8 - (*t)->size % 8);
    int32_t align_size = (*t)->size;
    // create data
    (*t)->data = (float_t*) malloc(sizeof(float_t) * align_size);
    for (int i = 0; i < align_size; ++i)
        (*t)->data[i] = val;
    return 0;
}

int32_t delete_float_tensor(FloatTensor **t)
{
    if (*t != NULL)
    {
        free((*t)->data);
        (*t)->data = NULL;
        free(*t);
        *t = NULL;
    }
    return 0;
}

int32_t create_uint8_tensor(UInt8Tensor **t, Shape shape, u_int8_t val)
{
    *t = (UInt8Tensor*) malloc(sizeof(UInt8Tensor));
    (*t)->shape = shape;
    (*t)->size = shape.n*shape.h*shape.w*shape.c;
    // 8 bit align
    // int32_t align_size = (*t)->size + (8 - (*t)->size % 8);
    int32_t align_size = (*t)->size;
    // create data
    (*t)->data = (u_int8_t*) malloc(sizeof(u_int8_t) * align_size);
    for (int i = 0; i < align_size; ++i)
        (*t)->data[i] = val;
    return 0;
}

int32_t delete_uint8_tensor(UInt8Tensor **t)
{
    if (*t != NULL)
    {
        free((*t)->data);
        (*t)->data = NULL;
        free(*t);
        *t = NULL;
    }
    return 0;
}

int32_t create_int8_tensor(Int8Tensor **t, Shape shape, int8_t val)
{
    *t = (Int8Tensor*) malloc(sizeof(Int8Tensor));
    (*t)->shape = shape;
    (*t)->size = shape.n*shape.h*shape.w*shape.c;
    // 8 bit align
    // int32_t align_size = (*t)->size + (8 - (*t)->size % 8);
    int32_t align_size = (*t)->size;
    // create data
    (*t)->data = (int8_t*) malloc(sizeof(int8_t) * align_size);
    for (int i = 0; i < align_size; ++i)
        (*t)->data[i] = val;
    return 0;
}

int32_t delete_int8_tensor(Int8Tensor **t)
{
    if (*t != NULL)
    {
        free((*t)->data);
        (*t)->data = NULL;
        free(*t);
        *t = NULL;
    }
    return 0;
}