#ifndef __CROSS_ENTROPY_LOSS_BACKWARD_CUDA_H__
#define __CROSS_ENTROPY_LOSS_BACKWARD_CUDA_H__

namespace op::cross_entropy_loss_backward::cuda {
typedef struct CrossEntropyLossBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b, const size_t N) const {
        float f_N = static_cast<float>(N);
        if constexpr (std::is_same_v<T, half2>) {
            half2 h2_N = __float2half2_rn(f_N);
            return __h2div(__hsub2(a, b), h2_N);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __hdiv(__hsub(a, b), __float2bfloat16(f_N));
        } else if constexpr (std::is_same_v<T, half>) {
            return __hdiv(__hsub(a, b), __float2half(f_N));
        } else if constexpr (std::is_same_v<T, float>) {
            return __fdiv_rn(__fsub_rn(a, b), f_N);
        } else {
            return (a - b) / static_cast<T>(N);
        }
    }
} CrossEntropyLossBackwardOp;
} // namespace op::cross_entropy_loss_backward::cuda

#endif // __CROSS_ENTROPY_LOSS_BACKWARD_CUDA_H__
