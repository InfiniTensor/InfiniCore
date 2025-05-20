#ifndef __INFINIOP_BANG_KERNEL_COMMON_H__
#define __INFINIOP_BANG_KERNEL_COMMON_H__

#include "cnnl.h"
#include "cnrt.h"

namespace device::bang::kernel {

inline __mlu_device__ size_t indexToReducedOffset(
    size_t flat_index,
    size_t ndim,
    const ptrdiff_t *broadcasted_strides,
    const ptrdiff_t *target_strides) {

    size_t res = 0;
    for (size_t i = 0; i < ndim; ++i) {
        res += flat_index / broadcasted_strides[i] * target_strides[i];
        flat_index %= broadcasted_strides[i];
    }
    return res;
}

inline __mlu_device__ size_t indexToOffset(
    size_t flat_index,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides) {

    size_t res = 0;
    for (size_t i = ndim; i-- > 0;) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
}

} // namespace device::bang::kernel

#endif
