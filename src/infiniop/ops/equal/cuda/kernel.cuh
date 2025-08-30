#ifndef __EQUAL_CUDA_H__
#define __EQUAL_CUDA_H__

namespace op::equal::cuda {
typedef struct EqualOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ bool operator()(const T &a, const T &b) const {
        return a == b;
    }
} EqualOp;
} // namespace op::equal::cuda

#endif // __EQUAL_CUDA_H__