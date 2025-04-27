#ifndef __INFINIOP_REDUCE_KUNLUN_H__
#define __INFINIOP_REDUCE_KUNLUN_H__

#include "../../devices/kunlun/kunlun_common.h"

namespace op::common_kunlun::reduce_op {

// Atomic add for reduce
inline __device__ void atomicAddF32(__shared_ptr__ float *ptr, float value) {
    int success = 1;
    while (success) {
        // SM2REG read 32bit data to register
        float a = SM2REG_atomic(ptr);
        a = a + value;
        success = REG2SM_atomic(ptr, a);
    }
}

inline __device__ void atomicMaxF32(__shared_ptr__ float *ptr, float value) {
    int success = 1;
    while (success) {
        float a = SM2REG_atomic(ptr);
        a = fmax(a, value);
        success = REG2SM_atomic(ptr, a);
    }
}

// Use 16 floats instruction to calculate reduce
// data_ptr is the pointer of LM
inline __device__ float sumSquaredF32(float *data_ptr, int count) {
    __local__ float acc_buf[16];
    int remain = count % 16;
    int offset_last = count - remain;
    int mask = lowerBitMask(remain - 1);
    // Load last 16 data
    float32x16_t v_last = vload_lm_float32x16_mz((data_ptr + offset_last), mask);
    // Do v_last * v_last
    v_last = vvmul_float32x16(v_last, v_last);
    // for every 16 float data
    for (int i = 0; i < offset_last; i += 16) {
        float32x16_t v_0 = vload_lm_float32x16_mz(data_ptr + i);
        // Do v_0 * v_0
        v_0 = vvmul_float32x16(v_0, v_0);
        // Add to v_last
        v_last = vvadd_float32x16(v_last, v_0);
    }
    vstore_lm_float32x16_mz(acc_buf, v_last);
    mfence();
    float res = 0.0f;
    for (int i = 0; i < 16; ++i) {
        res += acc_buf[i];
    }
    return res;
}

inline __device__ float sumF32(float *data_ptr, int count) {
    __local__ float acc_buf[16];
    int remain = count % 16;
    int offset_last = count - remain;
    int mask = lowerBitMask(remain - 1);
    // Load last 16 data
    float32x16_t v_last = vload_lm_float32x16_mz(data_ptr + offset_last, mask);
    // for every 16 float data
    for (int i = 0; i < offset_last; i += 16) {
        float32x16_t v_0 = vload_lm_float32x16_mz(data_ptr + i);
        // Add to v_last
        v_last = vvadd_float32x16(v_last, v_0);
    }
    vstore_lm_float32x16_mz(acc_buf, v_last);
    mfence();
    float res = 0.0f;
    for (int i = 0; i < 16; ++i) {
        res += acc_buf[i];
    }
    return res;
}

// Reduce max func
inline __device__ float maxF32(float *data_ptr, int count) {
    int remain = count % 16;
    int offset_last = count - remain;
    float res = -INF;
    for (int i = offset_last; i < count; i++) {
        res = fmax(res, *(data_ptr + i));
    }
    mfence();
    if (offset_last != 0) {
        __local__ float acc_buf[16];
        float32x16_t v_mv = vload_lm_float32x16_mz(data_ptr);
        // for every 16 float data
        for (int i = 16; i < offset_last; i += 16) {
            float32x16_t v_0 = vload_lm_float32x16_mz(data_ptr + i);
            v_mv = vvmax_float32x16_mz(v_mv, v_0);
        }
        vstore_lm_float32x16_mz(acc_buf, v_mv);
        mfence();
        for (int i = 0; i < 16; i++) {
            res = fmax(res, acc_buf[i]);
        }
    }
    return res;
}

} // namespace op::common_kunlun::reduce_op

#endif
