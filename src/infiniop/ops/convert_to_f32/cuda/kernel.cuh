#ifndef __CONVERT_TO_F32_CUDA_H__
#define __CONVERT_TO_F32_CUDA_H__

#include <type_traits>

namespace op::convert_to_f32::cuda {

struct ConvertToF32Op {
public:
    static constexpr size_t num_inputs = 1;

    template <typename Tout, typename Tin>
    __device__ __forceinline__ Tout operator()(const Tin &x) const {
        static_assert(std::is_same_v<Tout, float>, "convert_to_f32 output must be float");
        if constexpr (std::is_same_v<Tin, half>) {
            return __half2float(x);
        } else if constexpr (std::is_same_v<Tin, cuda_bfloat16>) {
            return __bfloat162float(x);
        } else if constexpr (std::is_same_v<Tin, float>) {
            return x;
        } else {
            return static_cast<float>(x);
        }
    }
};

} // namespace op::convert_to_f32::cuda

#endif // __CONVERT_TO_F32_CUDA_H__
