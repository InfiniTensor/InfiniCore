#ifndef __CROSSENTROPYLOSS_BACKWARD_CUDA_H__
#define __CROSSENTROPYLOSS_BACKWARD_CUDA_H__

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

namespace op::crossentropyloss_backward::cuda {

typedef struct CrossEntropyLossBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &probs, const T &target, const size_t N) const {
        float f_N = static_cast<float>(N);
        if constexpr (std::is_same_v<T, half2>) {
            half2 h2_N = __float2half2_rn(f_N);
            return __h2div(__hsub2(probs, target), h2_N);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __hdiv(__hsub(probs, target), __float2bfloat16(f_N));
        } else if constexpr (std::is_same_v<T, half>) {
            return __hdiv(__hsub(probs, target), __float2half(f_N));
        } else if constexpr (std::is_same_v<T, float>) {
            return __fdiv_rn(__fsub_rn(probs, target), f_N);
        } else {
            return (probs - target) / static_cast<T>(N);
        }
    }
} CrossEntropyLossBackwardOp;

} // namespace op::crossentropyloss_backward::cuda

#endif // __CROSSENTROPYLOSS_BACKWARD_CUDA_H__