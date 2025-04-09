#ifndef __INFINIOP_REDUCE_BANG_H__
#define __INFINIOP_REDUCE_BANG_H__

#include "bang.h"
// 注意__mlu_func__修饰的函数默认是内联函数
namespace op::common_bang {

namespace reduce_op {

__mlu_func__ void ComputeInternal(float *src, float *dest_sum_final, int max_num) {
    constexpr int reduce_num = 128 / sizeof(float);
    __bang_sumpool(
        dest_sum_final,
        src,
        /*channel*/ reduce_num,
        /*height*/ 1,
        /*width*/ max_num / reduce_num,
        /*kernel_height*/ 1,
        /*kernel_width*/ max_num / reduce_num,
        /*stride_height*/ 1,
        /*stride_width*/ 1);
    __bang_reduce_sum(dest_sum_final, dest_sum_final, reduce_num); // 这个函数把长度为max_num的向量做了一个norm求和
} // temp_nram是一个字节数为128的数组

__mlu_func__ void ComputeSum(float *src, float *dest_sum_final, int max_num) {
    ComputeInternal(src, dest_sum_final, max_num);
}

__mlu_func__ void ComputeSum(half *src, float *dest_sum_final, int max_num) {
    __bang_half2float(
        (float *)src, src + max_num, max_num); // src[2 * max_num]
    ComputeInternal((float *)src, dest_sum_final, max_num);
}

__mlu_func__ void ComputeSum(bfloat16_t *src, float *dest_sum_final,
                             int max_num) {
    __bang_bfloat162float((float *)src, src + max_num, max_num); // src[2 * max_num]
    ComputeInternal((float *)src, dest_sum_final, max_num);
}

template <typename T>
__mlu_func__ float reduceSumSquare(const T *source, T *src, float *dest_sum_final, int dimsize, int max_num) {
    int remain = dimsize % max_num;
    int repeat = (dimsize - remain) / max_num; // 一次搬运max_num，搬运dimsize个元素需要的次数
    float global_sum = 0.0f;
    int offset = (sizeof(T) == 2 ? max_num : 0); // 这是为了后面使用float精度计算half数据做的处理

    for (int s = 0; s < repeat; s++) {
        __memcpy(src + offset, source + s * max_num, max_num * sizeof(T), GDRAM2NRAM); // 如果T = half，把数据存储到src[max_num:2MaxNum]
        __bang_mul(src + offset, src + offset, src + offset, max_num);                 // src = src * src

        ComputeSum(src, dest_sum_final, max_num);
        global_sum += dest_sum_final[0];
    }
    if (remain) {
        __bang_write_zero(src, max_num + offset);
        __memcpy(src + offset, source + repeat * max_num, remain * sizeof(T), GDRAM2NRAM);
        __bang_mul(src + offset, src + offset, src + offset, remain); // src = src * src

        ComputeSum(src, dest_sum_final, max_num);
        global_sum += dest_sum_final[0];
    }
    return global_sum;

} // dest_sum_final是一个字节数为128的数组

template <typename T>
__mlu_func__ float reduceSum(const T *source, T *src, float *dest_sum_final, int dimsize, int max_num) {
    int remain = dimsize % max_num;
    int repeat = (dimsize - remain) / max_num; // 一次搬运max_num，搬运dimsize个元素需要的次数
    float global_sum = 0.0f;
    int offset = (sizeof(T) == 2 ? max_num : 0); // 这是为了后面使用float精度计算half数据做的处理

    for (int s = 0; s < repeat; s++) {
        __memcpy(src + offset, source + s * max_num, max_num * sizeof(T), GDRAM2NRAM); // 如果T = half，把数据存储到src[max_num:2MaxNum]

        ComputeSum(src, dest_sum_final, max_num);
        global_sum += dest_sum_final[0];
    }
    if (remain) {
        __bang_write_zero(src, max_num + offset);
        __memcpy(src + offset, source + repeat * max_num, remain * sizeof(T), GDRAM2NRAM);
        ComputeSum(src, dest_sum_final, max_num);

        global_sum += dest_sum_final[0];
    }
    return global_sum;

} // dest_sum_final是一个字节数为128的数组

} // namespace reduce_op

} // namespace op::common_bang

#endif
