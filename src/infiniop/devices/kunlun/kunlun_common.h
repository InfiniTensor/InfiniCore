#ifndef __INFINIOP_KUNLUN_COMMON_H__
#define __INFINIOP_KUNLUN_COMMON_H__

// This header file will only be include by .xpu file
#include "kunlun_type.h"
#include "xpu/kernel/xtdk.h"
#include "xpu/kernel/xtdk_math.h"
#include "xpu/kernel/xtdk_simd.h"
#include "xpu/runtime.h"
#include <utility>
#if !defined(__xpu__) || defined(__xpu_on_host__)
#include_next <assert.h>
#else
#define assert(x)
#endif

// Get mask for kunlun xpu 512bit register calculation
// if data is not enough to 512bit, padding zero and use
// mask to identify real data
// 0 - i bit 1, others 0
inline __device__ float lowerBitMask(int i) {
    return (1 << (i + 1)) - 1;
}

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
namespace op::kunlun_common {

inline __device__ kunlun_ptrdiff_t indexToReducedOffset(
    kunlun_ptrdiff_t flat_index,
    kunlun_size_t ndim,
    _global_ptr_ kunlun_ptrdiff_t *broadcasted_strides,
    _global_ptr_ kunlun_ptrdiff_t *target_strides) {
    kunlun_ptrdiff_t res = 0;

    __local__ kunlun_ptrdiff_t a[8];
    __local__ kunlun_ptrdiff_t b[8];

    for (kunlun_size_t i = 0; i < ndim; ++i) {
        GM2LM(broadcasted_strides + i, a + i, 1 * sizeof(kunlun_ptrdiff_t));
        GM2LM(target_strides + i, b + i, 1 * sizeof(kunlun_ptrdiff_t));
        res += flat_index / a[i] * b[i];
        flat_index %= a[i];
        mfence();
    }
    return res;
}

inline __device__ kunlun_ptrdiff_t indexToOffset(
    kunlun_ptrdiff_t flat_index,
    kunlun_size_t ndim,
    _global_ptr_ kunlun_size_t *shape,
    _global_ptr_ kunlun_ptrdiff_t *strides) {
    kunlun_ptrdiff_t res = 0;

    __local__ kunlun_ptrdiff_t b[8];
    __local__ kunlun_size_t c[8];

    for (kunlun_size_t i = ndim; i-- > 0;) {
        GM2LM(shape + i, c + i, 1 * sizeof(kunlun_size_t));
        GM2LM(strides + i, b + i, 1 * sizeof(kunlun_ptrdiff_t));

        res += (flat_index % c[i]) * b[i];
        flat_index /= c[i];
        mfence();
    }
    return res;
}

inline __device__ kunlun_ptrdiff_t getPaddedSize(
    kunlun_size_t ndim,
    _global_ptr_ kunlun_size_t *shape,
    _global_ptr_ kunlun_ptrdiff_t *pads) {
    kunlun_ptrdiff_t total_size = 1;

    __local__ kunlun_size_t c[8];
    __local__ kunlun_ptrdiff_t d[8];
    for (kunlun_size_t i = 0; i < ndim; ++i) {
        GM2LM(shape + i, c + i, 1 * sizeof(kunlun_size_t));
        GM2LM(pads + i, d + i, 1 * sizeof(kunlun_ptrdiff_t));

        total_size *= c[i] + (i < 2 ? 0 : 2 * d[i - 2]);
        mfence();
    }
    return total_size;
}

} // namespace op::kunlun_common

// TODO: atomicAddF16
// TODO: atomicAddI8
#endif
