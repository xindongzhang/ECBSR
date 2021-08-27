#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <stdio.h>
#include <math.h>
#include "tensor.h"


typedef enum OperationType
{
    CONV,
    PAD,
    ADD,
    MUL,
    SUB,
    RELU,
} OperationType;

typedef enum ArithmeticType
{
    FP32,
    INT8,
    UINT8,
} ArithmeticType;

typedef struct OperationConfig
{
    Shape inp_shape;
    Shape out_shape;
    OperationType operation_type;
    ArithmeticType arithmetic_type;
    int32_t thread_nums;

    int32_t scale;
    int32_t stride;
    int32_t pad_l;
    int32_t pad_r;
    int32_t pad_t;
    int32_t pad_d;
    int32_t dilation;
    int32_t kernel_size;
    int32_t groups;
    int32_t with_bias;
} OperationConfig;

#endif