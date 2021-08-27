#include "activation.h"

int32_t float_activation_relu(FloatTensor *out, FloatTensor *inp, OperationConfig config)
{
    for (int32_t i = 0; i < inp->size; ++i)
        out->data[i] = inp->data[i] * (inp->data[i] > 0.0);
    return 0;
}