#include "convolution.h"

int32_t float_conv3x3s1p1(
    FloatTensor *out, 
    FloatTensor *inp, 
    FloatTensor *ker, 
    FloatTensor *bias, 
    OperationConfig)
{
    for (int32_t oc = 0; oc < out->shape.c; ++oc)
    {
        float_t *out0 = out->data + oc * out->shape.h * out->shape.w;
        float_t *ker0 = ker->data;

        // fill bias
        float_t bias0 = bias? bias->data[oc]: 0.0f;
        int32_t total = out->shape.c * out->shape.h * out->shape.c;
        for (int32_t t = 0; t < total; ++t)
            out0[t] = bias0;

        // perform conv3x3
        for (int32_t ic = 0; ic < inp->shape.c; ++ic)
        {
            float_t *out0_ptr = out0 + 0 * out->shape.w;
            float_t *out1_ptr = out0 + 1 * out->shape.w;

            float_t *inp_ptr = inp->data + ic * inp->shape.h * inp->shape.w;

            float_t *inp0 = inp_ptr + inp->shape.w * 0;
            float_t *inp1 = inp_ptr + inp->shape.w * 1;
            float_t *inp2 = inp_ptr + inp->shape.w * 2;
            float_t *inp3 = inp_ptr + inp->shape.w * 3;

            int32_t oh = 0;
            for (; oh < out->shape.h - 1; oh += 2)
            {
                for (int32_t ow = 0; ow < out->shape.w; ++ow)
                {
                    float_t sum0 = 0.0f;
                    float_t sum1 = 0.0f;

                    sum0 += inp0[0] * ker0[0];
                    sum0 += inp0[1] * ker0[1];
                    sum0 += inp0[2] * ker0[2];
                    sum0 += inp1[0] * ker0[3];
                    sum0 += inp1[1] * ker0[4];
                    sum0 += inp1[2] * ker0[5];
                    sum0 += inp2[0] * ker0[6];
                    sum0 += inp2[1] * ker0[7];
                    sum0 += inp2[2] * ker0[8];         

                    sum1 += inp1[0] * ker0[0];
                    sum1 += inp1[1] * ker0[1];
                    sum1 += inp1[2] * ker0[2];
                    sum1 += inp2[0] * ker0[3];
                    sum1 += inp2[1] * ker0[4];
                    sum1 += inp2[2] * ker0[5];
                    sum1 += inp3[0] * ker0[6];
                    sum1 += inp3[1] * ker0[7];
                    sum1 += inp3[2] * ker0[8];

                    *(out0_ptr++) += sum0;
                    *(out1_ptr++) += sum1;

                    inp0++;
                    inp1++;
                    inp2++;
                    inp3++;      
                }

                inp0 += (2 + inp->shape.w);
                inp1 += (2 + inp->shape.w);
                inp2 += (2 + inp->shape.w);
                inp3 += (2 + inp->shape.w);

                out0_ptr += out->shape.w;
                out1_ptr += out->shape.w;
            }

            for (; oh < out->shape.h; ++oh)
            {
                for (int32_t ow = 0; ow < out->shape.w; ++ow)
                {
                    float_t sum0 = 0.0f;
                    
                    sum0 += inp0[0] * ker0[0];
                    sum0 += inp0[1] * ker0[1];
                    sum0 += inp0[2] * ker0[2];
                    sum0 += inp1[0] * ker0[3];
                    sum0 += inp1[1] * ker0[4];
                    sum0 += inp1[2] * ker0[5];
                    sum0 += inp2[0] * ker0[6];
                    sum0 += inp2[1] * ker0[7];
                    sum0 += inp2[2] * ker0[8];

                    *(out0_ptr++) += sum0;

                    inp0++;
                    inp1++;
                    inp2++;
                }
                inp0 += 2;
                inp1 += 2;
                inp2 += 2;
            }
            ker0 += 9;
        }
    }
    return 0;
}