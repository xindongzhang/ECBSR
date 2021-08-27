#include "padding.h"

int32_t float_padding(FloatTensor *out, FloatTensor *inp, OperationConfig config)
{
    int32_t pad_t = config.pad_t;
    int32_t pad_d = config.pad_d;
    int32_t pad_l = config.pad_l;
    int32_t pad_r = config.pad_r;

    for (int32_t n = 0; n < out->shape.n; ++n)
    {
        for (int32_t c = 0; c < out->shape.c; ++c)
        {
            for (int32_t h = pad_t; h < out->shape.h - pad_d; ++h)
            {
                int32_t oidx = n * out->shape.c * out->shape.h * out->shape.w + \
                               c * out->shape.h * out->shape.w + \
                               h * out->shape.w + \
                               pad_l;
                int32_t iidx = n * inp->shape.c * inp->shape.h * inp->shape.w + \
                               c * inp->shape.h * inp->shape.w + \
                               (h - pad_t) * inp->shape.w + \
                               0;
                memcpy(&out->data[oidx], &inp->data[iidx], sizeof(float_t) * inp->shape.w);
            }
        }
    }

    return 0;
}