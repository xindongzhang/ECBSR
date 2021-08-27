#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <omp.h>
#include <emmintrin.h>
#include "config.h"

int32_t float_activation_relu(FloatTensor *out, FloatTensor *inp, OperationConfig);

#endif