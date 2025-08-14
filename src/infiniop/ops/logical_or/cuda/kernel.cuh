#ifndef __LOGICAL_OR_CUDA_H__
#define __LOGICAL_OR_CUDA_H__

namespace op::logical_or::cuda {
typedef struct LogicalOrOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a || b;
    }
} LogicalOrOp;
} // namespace op::logical_or::cuda

#endif // __LOGICAL_OR_CUDA_H__
