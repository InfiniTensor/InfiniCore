#ifndef SIGMOID_BACKWARD_CUDA_H
#define SIGMOID_BACKWARD_CUDA_H

namespace op::sigmoid_backward::cuda {
typedef struct SigmoidBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &input, const T &grad_output) const {
        T sigmoid_val;

        if constexpr (std::is_same_v<T, half2>) {
            // For half2, process each component
            half2 one = __float2half2_rn(1.0f);
            half2 neg_input = __hneg2(input);
            half2 exp_neg_input = h2exp(neg_input);
            sigmoid_val = __h2div(one, __hadd2(one, exp_neg_input));
            return __hmul2(__hmul2(grad_output, sigmoid_val), __hsub2(one, sigmoid_val));
        } else if constexpr (std::is_same_v<T, half>) {
            half one = __float2half(1.0f);
            // half neg_input = __hneg(input);
            // half exp_neg_input = hexp(neg_input);
            // sigmoid_val = __hdiv(one, __hadd(one, exp_neg_input));
            sigmoid_val = sigmoid(input);
            return __hmul(__hmul(grad_output, sigmoid_val), __hsub(one, sigmoid_val));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            cuda_bfloat16 one = __float2bfloat16(1.0f);
            // cuda_bfloat16 neg_input = __hneg(input);
            // cuda_bfloat16 exp_neg_input = hexp(neg_input);
            // sigmoid_val = __hdiv(one, __hadd(one, exp_neg_input));
            sigmoid_val = sigmoid(input);
            return __hmul(__hmul(grad_output, sigmoid_val), __hsub(one, sigmoid_val));
        } else if constexpr (std::is_same_v<T, float>) {
            sigmoid_val = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-input)));
            return __fmul_rn(__fmul_rn(grad_output, sigmoid_val), __fsub_rn(1.0f, sigmoid_val));
        } else if constexpr (std::is_same_v<T, double>) {
            sigmoid_val = 1.0 / (1.0 + exp(-input));
            return grad_output * sigmoid_val * (1.0 - sigmoid_val);
        } else {
            // Fallback for other types
            sigmoid_val = static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-input));
            return grad_output * sigmoid_val * (static_cast<T>(1.0) - sigmoid_val);
        }
    }

private:
    __device__ __forceinline__ half sigmoid(const half &x) const {
        return __float2half(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(__half2float(-x)))));
    }
    __device__ __forceinline__ cuda_bfloat16 sigmoid(const cuda_bfloat16 &x) const {
        return __float2bfloat16(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(__bfloat162float(-x)))));
    }
} SigmoidBackwardOp;
} // namespace op::sigmoid_backward::cuda

#endif // SIGMOID_BACKWARD_CUDA_H
