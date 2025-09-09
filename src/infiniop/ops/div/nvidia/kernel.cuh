#ifndef __DIV_CUDA_H__
#define __DIV_CUDA_H__

namespace op::div::cuda {
typedef struct DivOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __h2div(a, b);
        } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
            return __hdiv(a, b);
        } else if constexpr (std::is_same_v<T, float>) {
            return __fdiv_rn(a, b);
        } else {
            return a / b;
        }
    }
} DivOp;
} // namespace op::div::cuda

#endif // __DIV_CUDA_H__
