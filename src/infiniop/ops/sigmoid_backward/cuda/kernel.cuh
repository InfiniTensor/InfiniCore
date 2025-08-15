#ifndef __SIGMOID_BACKWARD_CUDA_H__
#define __SIGMOID_BACKWARD_CUDA_H__

#include "../../../../utils/custom_types.h"

// Forward declarations of device fp16 conversion functions
__device__ __forceinline__ float device_f16_to_f32(fp16_t val);
__device__ __forceinline__ fp16_t device_f32_to_f16(float val);

// Forward declarations of device bf16 conversion functions
__device__ __forceinline__ float device_bf16_to_f32(bf16_t val);
__device__ __forceinline__ bf16_t device_f32_to_bf16(float val);

namespace op::sigmoid_backward::cuda {

// 高精度sigmoid函数实现
template<typename T>
__device__ __forceinline__ T sigmoid_func(T x) {
    if constexpr (std::is_same_v<T, half>) {
        // 对于half类型，使用内置函数
        return __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(x))));
    } else if constexpr (std::is_same_v<T, half2>) {
        // 对于half2类型
        half2 one = __float2half2_rn(1.0f);
        return __h2div(one, __hadd2(one, h2exp(__hneg2(x))));
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        // 对于bfloat16，转换为float计算以提高精度
        float x_float = __bfloat162float(x);
        float result = 1.0f / (1.0f + expf(-x_float));
        return __float2bfloat16(result);
    } else if constexpr (std::is_same_v<T, float>) {
        return 1.0f / (1.0f + expf(-x));
    } else if constexpr (std::is_same_v<T, fp16_t>) {
        // For fp16_t, convert to float for calculation
        float x_float = device_f16_to_f32(x);
        float result = 1.0f / (1.0f + expf(-x_float));
        return device_f32_to_f16(result);
    } else if constexpr (std::is_same_v<T, bf16_t>) {
        // For bf16_t, convert to float for calculation
        float x_float = device_bf16_to_f32(x);
        float result = 1.0f / (1.0f + expf(-x_float));
        return device_f32_to_bf16(result);
    } else {
        return static_cast<T>(1.0) / (static_cast<T>(1.0) + ::exp(-x));
    }
}

typedef struct SigmoidBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    
     template <typename T>
    __device__ __forceinline__ T operator()(const T &input, const T &grad_output) const {
        if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // 高精度版本：使用double作为中间计算类型
            float input_float = __bfloat162float(input);
            float grad_output_float = __bfloat162float(grad_output);
            
            double input_double = static_cast<double>(input_float);
            double grad_output_double = static_cast<double>(grad_output_float);
            
            double sigmoid_val = 1.0 / (1.0 + ::exp(-input_double));
            double result = grad_output_double * sigmoid_val * (1.0 - sigmoid_val);
            
            return __float2bfloat16(static_cast<float>(result));
        } else if constexpr (std::is_same_v<T, fp16_t>) {
            // For fp16_t, convert to float for calculation
            float input_float = device_f16_to_f32(input);
            float grad_output_float = device_f16_to_f32(grad_output);
            float sigmoid_val = 1.0f / (1.0f + expf(-input_float));
            float result = grad_output_float * sigmoid_val * (1.0f - sigmoid_val);
            return device_f32_to_f16(result);
        } else if constexpr (std::is_same_v<T, bf16_t>) {
            // For bf16_t, convert to float for calculation
            float input_float = device_bf16_to_f32(input);
            float grad_output_float = device_bf16_to_f32(grad_output);
            float sigmoid_val = 1.0f / (1.0f + expf(-input_float));
            float result = grad_output_float * sigmoid_val * (1.0f - sigmoid_val);
            return device_f32_to_bf16(result);
        } else {
            // 对于其他类型，使用标准实现
            T sigmoid_val = sigmoid_func(input);
            T one_minus_sigmoid = static_cast<T>(1.0) - sigmoid_val;
            return grad_output * sigmoid_val * one_minus_sigmoid;
        }
    }
} SigmoidBackwardOp;


} // namespace op::sigmoid_backward::cuda

#endif // __SIGMOID_BACKWARD_CUDA_H__