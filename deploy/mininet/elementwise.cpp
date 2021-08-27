#include "elementwise.h"

int32_t float_elemt_add(FloatTensor *out, FloatTensor *left, FloatTensor *right, OperationConfig config)
{
    if (is_shape_equal(left->shape, right->shape))
    {
#if WITH_SSE
        int32_t iters  = ceil(out->size / 4.0);
        int32_t remain = out->size % 4; 
        for (int32_t i = 0; i < iters; ++i)
        {
            __m128* o_addr = (__m128*) &out->data[i*4];
            __m128* l_addr = (__m128*) &left->data[i*4];
            __m128* r_addr = (__m128*) &right->data[i*4];  
            *o_addr = _mm_add_ps(*l_addr, *r_addr);    
        }
        for (int32_t i = 0; i < remain; ++i)
            out->data[iters*4 + i] = left->data[iters*4 + i] + right->data[iters*4 + i];
#else
        // #pragma omp parallel for 
        for (int32_t i = 0; i < out->size; ++i)
            out->data[i] = left->data[i] + right->data[i];   
#endif
    }
    return 0;
}

int32_t float_elemt_mul(FloatTensor *out, FloatTensor *left, FloatTensor *right, OperationConfig config)
{
    if (is_shape_equal(left->shape, right->shape))
    {
#if WITH_SSE
        int32_t iters  = ceil(out->size / 4.0);
        int32_t remain = out->size % 4; 
        for (int32_t i = 0; i < iters; ++i)
        {
            __m128* o_addr = (__m128*) &out->data[i*4];
            __m128* l_addr = (__m128*) &left->data[i*4];
            __m128* r_addr = (__m128*) &right->data[i*4];  
            *o_addr = _mm_mul_ps(*l_addr, *r_addr);    
        }
        for (int32_t i = 0; i < remain; ++i)
            out->data[iters*4 + i] = left->data[iters*4 + i] * right->data[iters*4 + i];
#else
        // #pragma omp parallel for 
        for (int32_t i = 0; i < out->size; ++i)
            out->data[i] = left->data[i] * right->data[i];   
#endif
    }
    return 0;
}
int32_t float_elemt_sub(FloatTensor *out, FloatTensor *left, FloatTensor *right, OperationConfig config)
{
    if (is_shape_equal(left->shape, right->shape))
    {
#if WITH_SSE
        int32_t iters  = ceil(out->size / 4.0);
        int32_t remain = out->size % 4; 
        for (int32_t i = 0; i < iters; ++i)
        {
            __m128* o_addr = (__m128*) &out->data[i*4];
            __m128* l_addr = (__m128*) &left->data[i*4];
            __m128* r_addr = (__m128*) &right->data[i*4];  
            *o_addr = _mm_sub_ps(*l_addr, *r_addr);   
        }
        for (int32_t i = 0; i < remain; ++i)
            out->data[iters*4 + i] = left->data[iters*4 + i] - right->data[iters*4 + i];
#else
        // #pragma omp parallel for 
        for (int32_t i = 0; i < out->size; ++i)
            out->data[i] = left->data[i] - right->data[i];   
#endif
    }
    return 0;
}