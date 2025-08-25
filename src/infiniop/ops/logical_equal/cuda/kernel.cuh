#ifndef __LOGICAL_EQUAL_CUDA_H__
#define __LOGICAL_EQUAL_CUDA_H__

namespace op::logical_equal::cuda {
typedef struct LogicalEqualOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a == b;
    }
} LogicalEqualOp;
} // namespace op::logical_equal::cuda

#endif // __logical_equal_CUDA_H__
