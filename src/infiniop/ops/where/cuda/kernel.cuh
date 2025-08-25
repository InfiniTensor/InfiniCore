#ifndef __WHERE_CUDA_H__
#define __WHERE_CUDA_H__

namespace op::where::cuda {
typedef struct WhereOp {
public:
    static constexpr size_t num_inputs = 3;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b, const bool &cond) const {
        return cond ? a : b;
    }
} WhereOp;
} // namespace op::where::cuda

#endif // __WHERE_CUDA_H__
