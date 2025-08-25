#ifndef __RELU_BACKWARD_CUDA_H__
#define __RELU_BACKWARD_CUDA_H__

namespace op::relu_backward::cuda {
typedef struct ReluBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        T zero{0};
        if (a > zero) {
            return b;
        } else {
            return zero;
        }
    }
} ReluBackwardOp;
} // namespace op::relu_backward::cuda

#endif // __RELU_BACKWARD_CUDA_H__
