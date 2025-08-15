#define INFINIOP_METAX_KERNEL __global__ void

// Posible maximum number of threads per block for METAX architectures
// Used for picking correct kernel launch configuration
#define METAX_BLOCK_SIZE_1024 1024
#define METAX_BLOCK_SIZE_512 512

#define CHECK_METAX(API) CHECK_INTERNAL(API, hcSuccess)

using cuda_bfloat16 = hpcc_bfloat16;
using cuda_bfloat162 = hpcc_bfloat162;

namespace device::metax {

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
} // namespace device::metax

__forceinline__ __device__ float
exp_(const float val) {
    return expf(val);
}

__forceinline__ __device__ long double
exp_(const long double val) {
    return exp(val);
}

__forceinline__ __device__ double
exp_(const double val) {
    return exp(val);
}

__forceinline__ __device__ __half
exp_(const __half x) {
    return hexp(x);
}

__forceinline__ __device__ __hpcc_bfloat16
exp_(const __hpcc_bfloat16 x) {
    return hexp(x);
}

__forceinline__ __device__ float
sin_(const float val) {
    return sinf(val);
}

__forceinline__ __device__ long double
sin_(const long double val) {
    return sin(val);
}

__forceinline__ __device__ double
sin_(const double val) {
    return sin(val);
}

__forceinline__ __device__ __half
sin_(const __half x) {
    return hsin(x);
}

__forceinline__ __device__ __hpcc_bfloat16
sin_(const __hpcc_bfloat16 x) {
    return hsin(x);
}

__forceinline__ __device__ float
cos_(const float val) {
    return cosf(val);
}

__forceinline__ __device__ long double
cos_(const long double val) {
    return cos(val);
}

__forceinline__ __device__ double
cos_(const double val) {
    return cos(val);
}

__forceinline__ __device__ __half
cos_(const __half x) {
    float x_float = __half2float(x);
    return __float2half(cosf(x_float));
}

__forceinline__ __device__ __hpcc_bfloat16
cos_(const __hpcc_bfloat16 x) {
    float x_float = __bfloat162float(x);
    return __float2bfloat16(cosf(x_float));
}

__forceinline__ __device__ float
tanh_(const float val) {
    return tanhf(val);
}

__forceinline__ __device__ long double
tanh_(const long double val) {
    return tanh(val);
}

__forceinline__ __device__ double
tanh_(const double val) {
    return tanh(val);
}

__forceinline__ __device__ __half
tanh_(const __half x) {
    float x_float = __half2float(x);
    return __float2half(tanhf(x_float));
}

__forceinline__ __device__ __hpcc_bfloat16
tanh_(const __hpcc_bfloat16 x) {
    float x_float = __bfloat162float(x);
    return __float2bfloat16(tanhf(x_float));
}
