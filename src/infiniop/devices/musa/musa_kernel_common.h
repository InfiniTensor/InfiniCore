#define INFINIOP_MUSA_KERNEL __global__ void
// Posible maximum number of threads per block for MUSA architectures
// Used for picking correct kernel launch configuration
#define MUSA_BLOCK_SIZE_1024 1024
#define MUSA_BLOCK_SIZE_512 512

#define CHECK_MUSA(API) CHECK_INTERNAL(API, musaSuccess)

namespace device::musa {

// return the memory offset of original tensor, given the flattened index of broadcasted tensor
__forceinline__ __device__ __host__ size_t
indexToReducedOffset(
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

// get the memory offset of the given element in a tensor given its flat index
__forceinline__ __device__ __host__ size_t
indexToOffset(
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
} // namespace device::musa

#ifdef ENABLE_MOORE_API
#include <musa_fp16.h>
__forceinline__ __device__ float
exp_(const float val) {
    return expf(val);
}

// <musa_fp16.h> not support expl
__forceinline__ __device__ long double
exp_(const long double val) {
    double d_val = static_cast<double>(val);
    double d_result = exp(d_val); // exp() is for double
    return static_cast<long double>(d_result);
}

__forceinline__ __device__ double
exp_(const double val) {
    return exp(val);
}

// <musa_fp16.h> not support hexp
__forceinline__ __device__ __half
exp_(const __half val) {
    float f_val = __half2float(val);
    float f_result = expf(f_val);
    return __float2half(f_result);
}

#endif
