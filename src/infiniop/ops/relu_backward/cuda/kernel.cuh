#ifndef __RELU_BACKWARD_CUDA_H__
#define __RELU_BACKWARD_CUDA_H__

namespace op::relu_backward::cuda {
typedef struct ReluBackWardOp {
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &input, const T &grad_output) const {
        auto dy = input > T(0) ? T(1) : T(0);
        if constexpr (std::is_same_v<T, half2> || std::is_same_v<T, cuda_bfloat162>) {
            return __hmul2(grad_output, dy);
        } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
            return __hmul(grad_output, dy);
        } else if constexpr (std::is_same_v<T, float>) {
            return __fmul_rn(grad_output, dy);
        } else {
            return grad_output * dy;
        }
    }
} ReluBackWardOp;

} // namespace op::relu_backward::cuda

#endif // __RELU_BACKWARD_CUDA_H__
