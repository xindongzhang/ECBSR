#ifndef __TENSOR_H__
#define __TENSRO_H__

#include <stdio.h>
#include <math.h>

#define TENSOR_FLOAT32 0
#define TENSOR_UINT8   1
#define TENSOR_INT8    2

typedef struct Shape
{
    int32_t n;
    int32_t c;
    int32_t h;
    int32_t w;
} Shape;

typedef struct FloatTensor
{
    /* data */
    Shape    shape;
    int32_t  size;
    float_t* data;
} FloatTensor;


typedef struct UInt8Tensor
{
    Shape     shape;
    int32_t   size;
    u_int8_t* data;
} UInt8Tensor;

typedef struct Int8Tensor
{
    Shape   shape;
    int32_t size;
    int8_t* data;
} Int8Tensor;

int32_t create_float_tensor(FloatTensor **t, Shape shape, float_t val);
int32_t delete_float_tensor(FloatTensor **t);

int32_t create_uint8_tensor(UInt8Tensor **t, Shape shape, u_int8_t val);
int32_t delete_uint8_tensor(UInt8Tensor **t);

int32_t create_int8_tensor(Int8Tensor **t, Shape, int8_t val);
int32_t delete_int8_tensor(Int8Tensor **t);

static int32_t is_shape_equal(Shape s0, Shape s1)
{
    if (
        s0.n == s1.n && s0.h == s1.h && 
        s0.w == s1.w && s0.c == s1.c
    ) 
        return 1;
    else 
        return 0;
};

#endif