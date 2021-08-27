#include "pixelshuffle.h"


int32_t float_pixelshuffle(FloatTensor *out, FloatTensor *inp, OperationConfig config)
{
    int32_t out_length3 = out->shape.c * out->shape.h * out->shape.w;
    int32_t out_length2 = out->shape.h * out->shape.w;
    int32_t out_length1 = out->shape.w;

    int32_t inp_length3 = inp->shape.c * inp->shape.h * inp->shape.w;
    int32_t inp_length2 = inp->shape.h * inp->shape.w;
    int32_t inp_length1 = inp->shape.w;

    int32_t scale1 = config.scale;
    int32_t scale2 = config.scale * config.scale;

    for (int32_t in = 0; in < inp->shape.n; ++in)
    {
        for (int32_t ic = 0; ic < inp->shape.c; ++ic)
        {
            for (int32_t ih = 0; ih < inp->shape.h; ++ih)
            {
                for (int32_t iw = 0; iw < inp->shape.w; ++iw)
                {
                    int32_t iidx = in * inp_length3 + \
                                   ic * inp_length2 + \
                                   ih * inp_length1 + \
                                   iw;
                    int32_t oidx = in * out_length3 + \
                                   (ic / scale2) * out_length2 + \
                                   (ih * scale1 + ic / scale1) * out_length1 + \
                                   (iw * scale1 + ic % scale1);
                    out->data[oidx] = inp->data[iidx];
                }
            }
        }
    }
    return 0;
}