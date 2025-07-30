#ifndef __EQUAL_CUDA_H__
#define __EQUAL_CUDA_H__

namespace op::equal::cuda {
typedef struct EqualOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename Tout, typename Ta, typename Tb>
    __device__ __forceinline__ Tout operator()(const Ta &a, const Tb &b) const {
        if constexpr (!std::is_same_v<Ta, Tb>) {
            printf("Ta and Tb must be the same type!\n");
            std::abort();
        }
        return a == b;
    }
} EqualOp;
} // namespace op::equal::cuda

#endif // __EQUAL_CUDA_H__
