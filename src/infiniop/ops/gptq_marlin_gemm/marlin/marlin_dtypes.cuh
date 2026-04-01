#ifndef _data_types_cuh
#define _data_types_cuh
#include "../sgl_kernel/utils.cuh"

#include "marlin.cuh"

namespace device::marlin {

template <typename scalar_t>
class ScalarType {
};

template <>
class ScalarType<__half> {
public:
    using scalar_t = __half;
    using scalar_t2 = fp16x2_t;

    // Matrix fragments for tensor core instructions; their precise layout is
    // documented here:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    using FragA = Vec<fp16x2_t, 4>;
    using FragB = Vec<fp16x2_t, 2>;
    using FragC = Vec<float, 4>;
    using FragS = Vec<fp16x2_t, 1>;
    using FragZP = Vec<fp16x2_t, 4>;

    static __device__ float inline num2float(const __half x) {
        return __half2float(x);
    }

    static __device__ fp16x2_t inline num2num2(const __half x) {
        return __half2half2(x);
    }

    static __device__ fp16x2_t inline nums2num2(const __half x1, const __half x2) {
        return __halves2half2(x1, x2);
    }

    static __host__ __device__ __half inline float2num(const float x) {
        return __float2half(x);
    }
};

template <>
class ScalarType<__nv_bfloat16> {
public:
    using scalar_t = __nv_bfloat16;
    using scalar_t2 = bf16x2_t;

    using FragA = Vec<bf16x2_t, 4>;
    using FragB = Vec<bf16x2_t, 2>;
    using FragC = Vec<float, 4>;
    using FragS = Vec<bf16x2_t, 1>;
    using FragZP = Vec<bf16x2_t, 4>;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    static __device__ float inline num2float(const __nv_bfloat16 x) {
        return __bfloat162float(x);
    }

    static __device__ bf16x2_t inline num2num2(const __nv_bfloat16 x) {
        return __bfloat162bfloat162(x);
    }

    static __device__ bf16x2_t inline nums2num2(const __nv_bfloat16 x1, const __nv_bfloat16 x2) {
        return __halves2bfloat162(x1, x2);
    }

    static __host__ __device__ __nv_bfloat16 inline float2num(const float x) {
        return __float2bfloat16(x);
    }
#endif
};

} // namespace device::marlin

#endif
