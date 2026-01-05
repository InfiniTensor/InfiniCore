#ifndef __MIN_CUDA_H__
#define __MIN_CUDA_H__

namespace op::min::cuda {
typedef struct MinOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __hmin2(a, b);
        } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
            return a < b ? a : b;
        } else if constexpr (std::is_same_v<T, float>) {
            return fminf(a, b);
        } else {
            return a < b ? a : b;
        }
    }
} MinOp;
} // namespace op::min::cuda

#endif // __MIN_CUDA_H__
