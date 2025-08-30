#ifndef __AND_CUDA_H__
#define __AND_CUDA_H__

namespace op::and_op::cuda {
typedef struct AndOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, bool>) {
            return a && b;
        } else {
            // For non-bool types, treat non-zero as true
            return (a != T(0)) && (b != T(0)) ? T(1) : T(0);
        }
    }
} AndOp;
} // namespace op::and_op::cuda

#endif // __AND_CUDA_H__