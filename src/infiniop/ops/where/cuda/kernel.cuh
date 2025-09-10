#ifndef __WHERE_CUDA_H__
#define __WHERE_CUDA_H__

namespace op::where::cuda {

typedef struct WhereOp {
public:
    static constexpr size_t num_inputs = 3;

    template <typename T, typename Tcond, typename Ta, typename Tb>
    __device__ __forceinline__ T operator()(const Tcond &condition, const Ta &a, const Tb &b) const {
        if constexpr (std::is_same_v<T, half2> || std::is_same_v<T, cuda_bfloat162>) {
            // For vectorized half types, apply element-wise selection
            return static_cast<T>(static_cast<bool>(condition) ? a : b);
        } else {
            // return condition ? a : b;
            return static_cast<T>(static_cast<bool>(condition) ? a : b);
        }
    }
} WhereOp;

} // namespace op::where::cuda

#endif // __WHERE_CUDA_H__