#ifndef __CROSSENTROPYLOSS_BACKWARD_METAX_H__
#define __CROSSENTROPYLOSS_BACKWARD_METAX_H__

#include <type_traits>

namespace op::crossentropyloss_backward::metax {

typedef struct CrossEntropyLossBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &probs, const T &target, const size_t N) const {
        float f_N = static_cast<float>(N);
        if constexpr (std::is_same_v<T, half>) {
            float f_probs = __half2float(probs);
            float f_target = __half2float(target);
            float result = (f_probs - f_target) / f_N;
            return __float2half(result);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float f_probs = __bfloat162float(probs);
            float f_target = __bfloat162float(target);
            float result = (f_probs - f_target) / f_N;
            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, float>) {
            return (probs - target) / f_N;
        } else {
            return (probs - target) / static_cast<T>(f_N);
        }
    }
} CrossEntropyLossBackwardOp;

} // namespace op::crossentropyloss_backward::metax

#endif // __CROSSENTROPYLOSS_BACKWARD_METAX_H__