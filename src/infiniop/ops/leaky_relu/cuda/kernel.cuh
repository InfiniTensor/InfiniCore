#ifndef __LEAKY_RELU_CUDA_H__
#define __LEAKY_RELU_CUDA_H__

#include "../../../../utils/custom_types.h"

// Forward declarations of device fp16 conversion functions
__device__ __forceinline__ float device_f16_to_f32(fp16_t val);
__device__ __forceinline__ fp16_t device_f32_to_f16(float val);

// Forward declarations of device bf16 conversion functions
__device__ __forceinline__ float device_bf16_to_f32(bf16_t val);
__device__ __forceinline__ bf16_t device_f32_to_bf16(float val);

namespace op::leaky_relu::cuda {

// Global variable to store negative slope
__device__ __constant__ float g_negative_slope = 0.01f;

typedef struct LeakyReLUOp {
public:
    static constexpr size_t num_inputs = 1;
    
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half>) {
            // For half type, use CUDA intrinsics
            half neg_slope_half = __float2half(g_negative_slope);
            half zero = __float2half(0.0f);
            return __hgt(x, zero) ? x : __hmul(x, neg_slope_half);
        } else if constexpr (std::is_same_v<T, half2>) {
            // For half2 type
            half2 neg_slope_half2 = __float2half2_rn(g_negative_slope);
            half2 zero = __float2half2_rn(0.0f);
            half2 mask = __hgt2(x, zero);
            half2 neg_part = __hmul2(x, neg_slope_half2);
            return __hadd2(__hmul2(x, mask), __hmul2(neg_part, __hsub2(__float2half2_rn(1.0f), mask)));
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            // For bfloat16, convert to float for calculation
            float x_float = __bfloat162float(x);
            float result = (x_float > 0.0f) ? x_float : x_float * g_negative_slope;
            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, fp16_t>) {
            // For fp16_t, convert to float for calculation
            float x_float = device_f16_to_f32(x);
            float result = (x_float > 0.0f) ? x_float : x_float * g_negative_slope;
            return device_f32_to_f16(result);
        } else if constexpr (std::is_same_v<T, bf16_t>) {
            // For bf16_t, convert to float for calculation
            float x_float = device_bf16_to_f32(x);
            float result = (x_float > 0.0f) ? x_float : x_float * g_negative_slope;
            return device_f32_to_bf16(result);
        } else if constexpr (std::is_same_v<T, float>) {
            // For float type
            return (x > 0.0f) ? x : x * g_negative_slope;
        } else {
            // For other types (double, etc.)
            return (x > static_cast<T>(0)) ? x : x * static_cast<T>(g_negative_slope);
        }
    }
} LeakyReLUOp;

// Function to set negative slope
void setNegativeSlope(float slope);

} // namespace op::leaky_relu::cuda

#endif // __LEAKY_RELU_CUDA_H__