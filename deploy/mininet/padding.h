#ifndef __PADDING_H__
#define __PADDING_H__

#include <omp.h>
#include <emmintrin.h>
#include <string.h>
#include "config.h"

int32_t float_padding(FloatTensor *out, FloatTensor *inp, OperationConfig config);

#endif