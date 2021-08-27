#include <iostream>
#include <thread>
#include <time.h>
#include "elementwise.h"
#include "convolution.h"
#include "padding.h"
#include "activation.h"
#include "pixelshuffle.h"

int32_t test_tensor(void)
{
    FloatTensor* t0;
    Shape shape0 = {
        .n = 1,
        .c = 1,
        .h = 3,
        .w = 3,
    };
    create_float_tensor(&t0, shape0, 0.0);
    for (int32_t i = 0; i < t0->size; ++i)
        printf("%d, %f\n", i, t0->data[i]);
    delete_float_tensor(&t0);
    if (t0 == NULL)
        printf("succesfully delete float tensor\n");

    UInt8Tensor* t1;
    Shape shape1 = {
        .n = 1,
        .c = 1,
        .h = 3,
        .w = 3,
    };
    create_uint8_tensor(&t1, shape1, 0);
    for (int32_t i = 0; i < t1->size; ++i)
        printf("%d, %d\n", i, t1->data[i]);
    delete_uint8_tensor(&t1);
    if (t1 == NULL)
        printf("succesfully delete uint8 tensor\n");

    Int8Tensor* t2;
    Shape shape2 = {
        .n = 1,
        .c = 1,
        .h = 3,
        .w = 3,
    };
    create_int8_tensor(&t2, shape2, 0);
    for (int32_t i = 0; i < t2->size; ++i)
        printf("%d, %d\n", i, t2->data[i]);
    delete_int8_tensor(&t2);
    if (t2 == NULL)
        printf("succesfully delete int8 tensor\n");
    
    return 0;
}


int32_t test_elementwise_ops(void)
{
    FloatTensor *t0_add, *t1_add, *t2_add;
    Shape shape = {
        .n = 1,
        .c = 1,
        .h = 3,
        .w = 3,
    };
    OperationConfig config;
    config.inp_shape = shape;
    config.out_shape = shape;
    config.operation_type = OperationType::ADD;

    create_float_tensor(&t0_add, shape, 0.0);
    create_float_tensor(&t1_add, shape, 2.0);
    create_float_tensor(&t2_add, shape, 1.0);

    clock_t start = clock();
    float_elemt_mul(t0_add, t1_add, t2_add, config);
    clock_t end   = clock();
    float_t duration = (float_t) (end - start) / CLOCKS_PER_SEC;

    for (int32_t i = 0; i < t0_add->size; ++i)
        printf("%d, %f\n", i, t0_add->data[i]);
    printf("elmt durations is: %f\n", duration);

    delete_float_tensor(&t0_add);
    delete_float_tensor(&t1_add);
    delete_float_tensor(&t2_add);

    return 0;
}


int32_t test_conv3x3(void)
{
    FloatTensor *out, *inp, *ker;
    Shape out_shape = {
        .n = 16,
        .c = 16,
        .h = 480,
        .w = 640,
    };
    Shape inp_shape = {
        .n = 16,
        .c = 16,
        .h = 482,
        .w = 642,
    };
    Shape ker_shape = {
        .n = 16,
        .c = 16,
        .h = 3,
        .w = 3,
    };

    OperationConfig config;
    config.kernel_size = 3;
    config.pad_d = 0;
    config.pad_l = 0;
    config.pad_r = 0;
    config.pad_t = 0;
    config.stride = 1;
    create_float_tensor(&out, out_shape, 0.0f);
    create_float_tensor(&inp, inp_shape, 1.0f);
    create_float_tensor(&ker, ker_shape, 1.0f);

    clock_t start = clock();
    float_conv3x3s1p1(out, inp, ker, NULL, config);
    clock_t end = clock();
    float_t duration = (float_t) (end - start) / CLOCKS_PER_SEC;

    // for (int32_t i = 0; i < out->size; ++i)
        // printf("%d, %f\n", i, out->data[i]);
    printf("conv3x3 durations is : %f\n", duration);
    printf("inp shape: (%d, %d, %d, %d)\n", inp->shape.n, inp->shape.c, inp->shape.h, inp->shape.w);

    delete_float_tensor(&out);
    delete_float_tensor(&inp);
    delete_float_tensor(&ker);

    return 0;
}


int32_t test_padding(void)
{
    FloatTensor *out, *inp;
    Shape out_shape = {
        .n = 1,
        .c = 16,
        .h = 482,
        .w = 642,
    };
    Shape inp_shape = {
        .n = 1,
        .c = 16,
        .h = 480,
        .w = 640,
    };
    OperationConfig config;
    config.pad_t = 1;
    config.pad_d = 1;
    config.pad_l = 1;
    config.pad_r = 1;
    create_float_tensor(&out, out_shape, 0.0);
    create_float_tensor(&inp, inp_shape, 1.0);

    clock_t start = clock();
    float_padding(out, inp, config);
    clock_t end = clock();
    float_t duration = (float_t) (end - start) / CLOCKS_PER_SEC;

    // for (int32_t i = 0; i < out->size; ++i)
        // printf("%d, %f\n", i, out->data[i]);
    printf("padding durations is %f\n", duration);

    delete_float_tensor(&out);
    delete_float_tensor(&inp);
}

int32_t test_padding_conv3x3(void)
{
    FloatTensor *out, *pad, *inp, *ker;
    // Shape out_shape = {
    //     .n = 1,
    //     .c = 16,
    //     .h = 240,
    //     .w = 360,
    // };
    // Shape pad_shape = {
    //     .n = 1,
    //     .c = 16,
    //     .h = 242,
    //     .w = 362,
    // };
    // Shape inp_shape = {
    //     .n = 1,
    //     .c = 16,
    //     .h = 240,
    //     .w = 360,
    // };
    // Shape ker_shape = {
    //     .n = 16,
    //     .c = 16,
    //     .h = 3,
    //     .w = 3,
    // };
    Shape out_shape = {
        .n = 1,
        .c = 16,
        .h = 480,
        .w = 640,
    };
    Shape pad_shape = {
        .n = 1,
        .c = 16,
        .h = 482,
        .w = 642,
    };
    Shape inp_shape = {
        .n = 1,
        .c = 16,
        .h = 480,
        .w = 640,
    };
    Shape ker_shape = {
        .n = 16,
        .c = 16,
        .h = 3,
        .w = 3,
    };
    OperationConfig config;
    config.kernel_size = 3;
    config.pad_d = 1;
    config.pad_l = 1;
    config.pad_r = 1;
    config.pad_t = 1;
    config.stride = 1;
    create_float_tensor(&out, out_shape, 0.0);
    create_float_tensor(&pad, pad_shape, 0.0);
    create_float_tensor(&inp, inp_shape, 1.0);
    create_float_tensor(&ker, ker_shape, 1.0);

    clock_t start = clock();
    float_padding(pad, inp, config);
    float_conv3x3s1p1(out, pad, ker, NULL, config);
    clock_t end = clock();
    float_t duration = (float_t) (end - start) / CLOCKS_PER_SEC;

    // for (int32_t i = 0; i < out->size; ++i)
        // printf("%d, %f\n", i, out->data[i]);
    printf("padding conv3x3 durations is : %f\n", duration);
    
    delete_float_tensor(&out);
    delete_float_tensor(&pad);
    delete_float_tensor(&inp);
    delete_float_tensor(&ker);

    return 0;
}

int32_t test_padding_conv3x3_relu(void)
{
    FloatTensor *out, *pad, *inp, *ker;
    Shape out_shape = {
        .n = 1,
        .c = 16,
        .h = 480,
        .w = 640,
    };
    Shape pad_shape = {
        .n = 1,
        .c = 16,
        .h = 482,
        .w = 642,
    };
    Shape inp_shape = {
        .n = 1,
        .c = 16,
        .h = 480,
        .w = 640,
    };
    Shape ker_shape = {
        .n = 16,
        .c = 16,
        .h = 3,
        .w = 3,
    };
    OperationConfig config;
    config.kernel_size = 3;
    config.pad_d = 1;
    config.pad_l = 1;
    config.pad_r = 1;
    config.pad_t = 1;
    config.stride = 1;
    create_float_tensor(&out, out_shape, 0.0);
    create_float_tensor(&pad, pad_shape, 0.0);
    create_float_tensor(&inp, inp_shape, 1.0);
    create_float_tensor(&ker, ker_shape, 1.0);

    clock_t start = clock();
    float_padding(pad, inp, config);
    float_conv3x3s1p1(out, pad, ker, NULL, config);
    float_activation_relu(out, out, config);
    clock_t end = clock();
    float_t duration = (float_t) (end - start) / CLOCKS_PER_SEC;

    // for (int32_t i = 0; i < out->size; ++i)
        // printf("%d, %f\n", i, out->data[i]);
    printf("padding conv3x3 relu durations is : %f\n", duration);
    
    delete_float_tensor(&out);
    delete_float_tensor(&pad);
    delete_float_tensor(&inp);
    delete_float_tensor(&ker);

    return 0;
}

int32_t test_pixelshuffle(void)
{
    FloatTensor *out, *inp;
    Shape inp_shape = {
        .n = 1,
        .c = 9,
        .h = 1,
        .w = 1
    };
    Shape out_shape = {
        .n = 1,
        .c = 1,
        .h = 3,
        .w = 3
    };
    OperationConfig config;
    config.scale = 3;
    create_float_tensor(&out, out_shape, 0.0);
    create_float_tensor(&inp, inp_shape, 1.0);

    for (int32_t i = 0; i < inp->size; ++i)
        inp->data[i] = i;

    clock_t start = clock();
    float_pixelshuffle(out, inp, config);
    clock_t end   = clock();
    float_t duration = (float_t) (end - start) / CLOCKS_PER_SEC;

    for (int32_t i = 0; i < out->size; ++i)
        printf("%d, %f\n", i, out->data[i]);

    printf("the pixelshuffle duration is : %f\n", duration);
    return 0;
}

int32_t test_padding_conv3x3_bias_relu(void)
{
    FloatTensor *out, *pad, *inp, *ker, *bias;
    Shape out_shape = {
        .n = 1,
        .c = 16,
        .h = 480,
        .w = 640,
    };
    Shape bias_shape = {
        .n = 1,
        .c = 16,
        .h = 1,
        .w = 1,
    };
    Shape pad_shape = {
        .n = 1,
        .c = 16,
        .h = 482,
        .w = 642,
    };
    Shape inp_shape = {
        .n = 1,
        .c = 16,
        .h = 480,
        .w = 640,
    };
    Shape ker_shape = {
        .n = 16,
        .c = 16,
        .h = 3,
        .w = 3,
    };
    OperationConfig config;
    config.kernel_size = 3;
    config.pad_d = 1;
    config.pad_l = 1;
    config.pad_r = 1;
    config.pad_t = 1;
    config.stride = 1;
    create_float_tensor(&out, out_shape, 0.0);
    create_float_tensor(&bias, bias_shape, 1.0);
    create_float_tensor(&pad, pad_shape, 0.0);
    create_float_tensor(&inp, inp_shape, 1.0);
    create_float_tensor(&ker, ker_shape, 1.0);

    clock_t start = clock();
    float_padding(pad, inp, config);
    float_conv3x3s1p1(out, pad, ker, bias, config);
    float_activation_relu(out, out, config);
    clock_t end = clock();
    float_t duration = (float_t) (end - start) / CLOCKS_PER_SEC;

    // for (int32_t i = 0; i < out->size; ++i)
        // printf("%d, %f\n", i, out->data[i]);
    printf("padding conv3x3 relu durations is : %f\n", duration);
    
    delete_float_tensor(&out);
    delete_float_tensor(&bias);
    delete_float_tensor(&pad);
    delete_float_tensor(&inp);
    delete_float_tensor(&ker);

    return 0;
}