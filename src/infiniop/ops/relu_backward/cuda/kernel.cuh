#ifndef RELU_BACKWARD_CUDA_H
#define RELU_BACKWARD_CUDA_H

namespace op::relu_backward::cuda {
typedef struct ReluBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &input, const T &grad_output) const {
        if constexpr (std::is_same_v<T, half2>) {
            half2 zero = __float2half2_rn(0.0f);
            half2 mask = __hgt2(input, zero);
            return __hmul2(grad_output, mask);
        } else if constexpr (std::is_same_v<T, half>) {
            half zero = __float2half(0.0f);
            return __hgt(input, zero) ? grad_output : zero;
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            cuda_bfloat16 zero = __float2bfloat16(0.0f);
            return __hgt(input, zero) ? grad_output : zero;
        } else if constexpr (std::is_same_v<T, float>) {
            return input > 0.0f ? grad_output : 0.0f;
        } else if constexpr (std::is_same_v<T, double>) {
            return input > 0.0 ? grad_output : 0.0;
        } else {
            return input > T(0) ? grad_output : T(0);
        }
    }
} ReluBackwardOp;
} // namespace op::relu_backward::cuda

#endif // RELU_BACKWARD_CUDA_H
