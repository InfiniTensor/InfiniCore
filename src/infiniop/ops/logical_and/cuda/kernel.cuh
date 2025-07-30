#ifndef __LOGICAL_AND_CUDA_H__
#define __LOGICAL_AND_CUDA_H__

namespace op::logical_and::cuda {
typedef struct LogicalAndOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a && b;
    }
} LogicalAndOp;
} // namespace op::logical_and::cuda

#endif // __logical_and_CUDA_H__
