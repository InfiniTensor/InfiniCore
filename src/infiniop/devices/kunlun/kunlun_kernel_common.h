#ifndef __INFINIOP_KUNLUN_KERNEL_COMMON_H__
#define __INFINIOP_KUNLUN_KERNEL_COMMON_H__

// This header file will only be include by .xpu file
#include "xpu/runtime.h"
#include <xpu/kernel/xtdk.h>
#include <xpu/kernel/xtdk_atomic_sm_xpu3.h>
#include <xpu/kernel/xtdk_bf16.h>
#include <xpu/kernel/xtdk_math.h>
#include <xpu/kernel/xtdk_simd.h>
#include <xpu/kernel/xtdk_trigonometric.h>

namespace device::kunlun::kernel {

#define SM_SIZE 40960

/**
 * @brief Define ptrdiff_t and size_t for kunlun xpu
 * ptrdiff_t is 32 bit, size_t is 32 bit in xpu kernel
 * We padding it into 64 bit for convience of DATACOPY
 */
typedef struct _ptrdiff_t {
    int32_t value;   // 32 bit
    int32_t padding; // 32 bit
} _ptrdiff_t;
// same as ptrdiff
typedef struct _size_t {
    uint32_t value;
    uint32_t padding;
} _size_t;

// Get mask for kunlun xpu 512bit register calculation
// if data is not enough to 512bit, padding zero and use
// mask to identify real data
// 0 - i bit 1, others 0
inline __device__ float lowerBitMask(int i) {
    return (1 << (i + 1)) - 1;
}

// Load len data from shared memory
template <typename T>
__device__ inline void loadsm(__shared_ptr__ const T *p, T *v, int len) {
    __builtin_memcpy(v, p, len * sizeof(T));
}

/**
 * @brief atomicAdd for kunlun xpu
 * @param ptr: pointer to shared memory
 * @param value: value to add
 */
template <typename T>
inline __device__ T atomicAdd(__shared_ptr__ T *ptr, T value) {
    T x = atomicadd(ptr, value);
    return x;
}

// Specialize atomicAdd for half - Fixed address space issue
template <>
inline __device__ half atomicAdd<half>(__shared_ptr__ half *ptr, half value) {
    ticket_lock_mix();
    // Load from shared memory to register first
    half old_val;
    __builtin_memcpy(&old_val, ptr, sizeof(half));
    float of = __half2float(old_val);
    float vf = __half2float(value);
    float sumf = of + vf;
    half sum = __float2half_rn(sumf);
    __builtin_memcpy(ptr, &sum, sizeof(half));
    mfence_sm();
    ticket_unlock_mix();
    return old_val;
}

// Specialize atomicAdd for bfloat16_t - Fixed address space issue
template <>
inline __device__ bfloat16_t atomicAdd<bfloat16_t>(__shared_ptr__ bfloat16_t *ptr, bfloat16_t value) {
    ticket_lock_mix();
    // Load from shared memory to register first
    bfloat16_t old_val;
    __builtin_memcpy(&old_val, ptr, sizeof(bfloat16_t));
    float of = __bfloat162float(old_val);
    float vf = __bfloat162float(value);
    float sumf = of + vf;
    bfloat16_t sum = __float2bfloat16_rn(sumf);
    __builtin_memcpy(ptr, &sum, sizeof(bfloat16_t));
    mfence_sm();
    ticket_unlock_mix();
    return old_val;
}

/**
 * @brief atomicMax for kunlun xpu - Fixed address space issue
 * @param ptr: pointer to shared memory
 * @param value: value to compare
 */
template <typename T>
inline __device__ T atomicMax(__shared_ptr__ T *ptr, T value) {
    ticket_lock_mix();
    // Load from shared memory to register first
    T old_val;
    __builtin_memcpy(&old_val, ptr, sizeof(T));
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        float of = __bfloat162float(old_val);
        float vf = __bfloat162float(value);
        float maxf = fmax(of, vf);
        bfloat16_t max_val = __float2bfloat16_rn(maxf);
        __builtin_memcpy(ptr, &max_val, sizeof(T));
    } else {
        T max_val = fmax(old_val, value);
        __builtin_memcpy(ptr, &max_val, sizeof(T));
    }
    mfence_sm();
    ticket_unlock_mix();
    return old_val;
}

/**
 * @brief Get real offset of input index
 * flat_index: flatten index input
 * ndim: dim of input tensor
 * shape: shape of input tensor
 * strides: strides of input tensor
 */
inline __device__ int indexToOffset(
    int flat_index,
    int ndim,
    const _size_t *shape,
    const _ptrdiff_t *strides) {

    int res = 0;
    for (int i = ndim; i-- > 0;) {
        res += (flat_index % shape[i].value) * strides[i].value;
        flat_index /= shape[i].value;
    }
    return res;
}

/**
 * @brief Get max of a array of local mem
 * @param data: pointer to local memory
 * @param len: length of array
 * @return max value
 */
template <typename T>
__inline__ __device__ T max(const T *data_ptr, size_t len) {
    T max_val = data_ptr[0];
    for (size_t i = 0; i < len; ++i) {
        max_val = fmax(max_val, data_ptr[i]);
    }
    return max_val;
}

// Use simd vector instruction to calculate max of a half array
template <>
__inline__ __device__ half max(const half *data_ptr, size_t len) {
    int remain = len % 32;
    int offset_last = len - remain;
    half res = data_ptr[0];
    for (int i = offset_last; i < len; i++) {
        res = fmax(res, *(data_ptr + i));
    }
    mfence();
    if (offset_last != 0) {
        __local__ half acc_buf[32];
        float16x32_t v_mv = vload_lm_float16x32_mz(data_ptr);
        // for every 16 float data
        for (int i = 32; i < offset_last; i += 32) {
            float16x32_t v_0 = vload_lm_float16x32_mz(data_ptr + i);
            v_mv = vvmax_float16x32_mz(v_mv, v_0);
        }
        vstore_lm_float16x32_mz(acc_buf, v_mv);
        mfence();
        for (int i = 0; i < 32; i++) {
            res = fmax(res, acc_buf[i]);
        }
    }
    return res;
}

// Use simd vector instruction to calculate max of a half array
template <>
__inline__ __device__ float max(const float *data_ptr, size_t len) {
    int remain = len % 16;
    int offset_last = len - remain;
    float res = data_ptr[0];
    for (int i = offset_last; i < len; i++) {
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

// Helper function to load from shared memory to register (for half types)
inline __device__ half loadFromShared(__shared_ptr__ const half *ptr) {
    half val;
    __builtin_memcpy(&val, ptr, sizeof(half));
    return val;
}

// Helper function to store to shared memory from register (for half types)
inline __device__ void storeToShared(__shared_ptr__ half *ptr, half val) {
    __builtin_memcpy(ptr, &val, sizeof(half));
}

// Helper function to load from shared memory to register (for bfloat16 types)
inline __device__ bfloat16_t loadFromShared(__shared_ptr__ const bfloat16_t *ptr) {
    bfloat16_t val;
    __builtin_memcpy(&val, ptr, sizeof(bfloat16_t));
    return val;
}

// Helper function to store to shared memory from register (for bfloat16 types)
inline __device__ void storeToShared(__shared_ptr__ bfloat16_t *ptr, bfloat16_t val) {
    __builtin_memcpy(ptr, &val, sizeof(bfloat16_t));
}

} // namespace device::kunlun::kernel

#endif // __INFINIOP_KUNLUN_KERNEL_COMMON_H__
