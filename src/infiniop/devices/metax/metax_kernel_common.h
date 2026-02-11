#define INFINIOP_METAX_KERNEL __global__ void

#ifdef ENABLE_METAX_MC_API
#include <maca_fp8.h>
#else
#include <common/hpcc_fp8.h>
#include <common/hpcc_fp16.h>  // For hexp(__half)
#include <common/hpcc_bfloat16.h>  // For hexp(__hpcc_bfloat16)
#endif

// Posible maximum number of threads per block for METAX architectures
// Used for picking correct kernel launch configuration
#define METAX_BLOCK_SIZE_1024 1024
#define METAX_BLOCK_SIZE_512 512

#define CHECK_METAX(API) CHECK_INTERNAL(API, hcSuccess)

using cuda_bfloat16 = hpcc_bfloat16;
using cuda_bfloat162 = hpcc_bfloat162;
using cuda_fp8_e4m3 = __hpcc_fp8_e4m3;

namespace device::metax {

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
#ifdef __HPCCCC__
    return hexp(x);
#else
    // When not using HPCC compiler, convert to float, compute exp, convert back
    float f_val = __half2float(x);
    float f_result = expf(f_val);
    return __float2half(f_result);
#endif
}

__forceinline__ __device__ __hpcc_bfloat16
exp_(const __hpcc_bfloat16 x) {
#ifdef __HPCCCC__
    return hexp(x);
#else
    // When not using HPCC compiler, convert to float, compute exp, convert back
    // Use __bfloat162float from HPCC header for conversion
    float f_val = __bfloat162float(x);
    float f_result = expf(f_val);
    return __float2bfloat16(f_result);
#endif
}
