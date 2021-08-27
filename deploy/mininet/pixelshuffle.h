#ifndef __PIXELSHUFFLE_H__
#define __PIXELSHUFFLE_H__

#include <omp.h>
#include <emmintrin.h>
#include "config.h"

int32_t float_pixelshuffle(FloatTensor *out, FloatTensor *inp, OperationConfig config);

#endif