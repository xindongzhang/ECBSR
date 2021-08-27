#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

#include <omp.h>
#include <emmintrin.h>
#include "config.h"

int32_t float_conv3x3s1p1(
    FloatTensor *out, 
    FloatTensor *inp, 
    FloatTensor *ker, 
    FloatTensor *bias, 
    OperationConfig
);

#endif