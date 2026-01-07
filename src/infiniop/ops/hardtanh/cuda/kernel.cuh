#ifndef __HARDTANH_CUDA_H__
#define __HARDTANH_CUDA_H__

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

namespace op::hardtanh::cuda {

typedef struct HardTanhOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, float min_val, float max_val) const {
        if constexpr (std::is_same_v<T, half2>) {
            // half2 向量化优化：一次处理两个 FP16
            float2 x_f2 = __half22float2(x);
            x_f2.x = fminf(max_val, fmaxf(min_val, x_f2.x));
            x_f2.y = fminf(max_val, fmaxf(min_val, x_f2.y));
            return __float22half2_rn(x_f2);
            
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16：转 float 计算再转回
            float x_f = __bfloat162float(x);
            return __float2bfloat16(fminf(max_val, fmaxf(min_val, x_f)));
            
        } else if constexpr (std::is_same_v<T, half>) {
            // FP16：转 float 计算再转回
            float x_f = __half2float(x);
            return __float2half(fminf(max_val, fmaxf(min_val, x_f)));
            
        } else if constexpr (std::is_same_v<T, float>) {
            // FP32：直接使用内置 fminf/fmaxf
            return fminf(max_val, fmaxf(min_val, x));
            
        } else if constexpr (std::is_same_v<T, double>) {
            // FP64：使用双精度 fmin/fmax
            return fmin((double)max_val, fmax((double)min_val, x));
        }
    }
} HardTanhOp;

} // namespace op::hardtanh::cuda

#endif // __HARDTANH_CUDA_H__
