#ifndef __LOGICAL_AND_CUDA_H__
#define __LOGICAL_AND_CUDA_H__

namespace op::logical_and::cuda {
typedef struct LogicalAndOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename Tout, typename Ta, typename Tb>
    __device__ __forceinline__ Tout operator()(const Ta &a, const Tb &b) const {
        if constexpr (!std::is_same_v<Ta, Tb>) {
            printf("Ta and Tb must be the same type!\n");
            std::abort();
        }
        return a && b;
    }
} LogicalAndOp;
} // namespace op::logical_and::cuda

#endif // __LOGICAL_AND_CUDA_H__
