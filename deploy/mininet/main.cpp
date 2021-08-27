#include "test.h"

int main(void)
{
    // int32_t t_tensor_flag = test_tensor();
    // int32_t t_elemt_ops_flag = test_elementwise_ops();
    // int32_t t_padding_op_flag = test_padding();
    // int32_t t_conv3x3_op_flag = test_conv3x3();
    // int32_t t_pad_conv3x3_flag = test_padding_conv3x3();
    // int32_t t_pad_conv3x3_relu_flag = test_padding_conv3x3_relu();
    int32_t t_pad_conv3x3_bias_relu_flag = test_padding_conv3x3_bias_relu();
    // int32_t t_pixelshuffle_flag = test_pixelshuffle();
    return 0;
}