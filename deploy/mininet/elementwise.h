#ifndef __ELEMENTWISE_H__
#define __ELEMENTWISE_H__

#include <omp.h>
#include <emmintrin.h>
#include "config.h"

int32_t float_elemt_add(FloatTensor *out, FloatTensor *left, FloatTensor *right, OperationConfig config);
int32_t float_elemt_mul(FloatTensor *out, FloatTensor *left, FloatTensor *right, OperationConfig config);
int32_t float_elemt_sub(FloatTensor *out, FloatTensor *left, FloatTensor *right, OperationConfig config);

#endif