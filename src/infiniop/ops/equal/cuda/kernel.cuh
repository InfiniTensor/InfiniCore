#ifndef __EQUAL_CUDA_H__
#define __EQUAL_CUDA_H__

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

namespace op::equal::cuda {

typedef struct EqualOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        // Case 1: Half2 (FP16 向量化)
        if constexpr (std::is_same_v<T, half2>) {
            // __heq2 返回 1.0 (True) 或 0.0 (False) 的 half2 格式
            return __heq2(a, b);
        } 
        // Case 2: Half (FP16 标量)
        else if constexpr (std::is_same_v<T, half>) {
            // __heq 返回 bool，需要强转回 T (1.0/0.0)
            return static_cast<T>(__heq(a, b));
        }
        // Case 3: BFloat16
        else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16 比较通常转 float 或使用 intrinsic
             return static_cast<T>(a == b);
        }
        // Case 4: Float / Int
        else {
            // 标准比较，结果转为 T (1/0)
            // 注意：Elementwise 框架通常希望返回 T 类型以便写入 Output
            return static_cast<T>(a == b);
        }
    }

    template <typename Tout, typename Tin0, typename Tin1>
    __device__ __forceinline__ Tout operator()(const Tin0 &a, const Tin1 &b) const {
        static_assert(std::is_same_v<Tin0, Tin1>, "EqualOp expects identical input dtypes");
        if constexpr (std::is_same_v<Tin0, half2>) {
            static_assert(!std::is_same_v<Tin0, half2>, "half2 is not supported for mixed output dtype");
        } else if constexpr (std::is_same_v<Tin0, half>) {
            return static_cast<Tout>(__heq(a, b));
        } else {
            return static_cast<Tout>(a == b);
        }
    }
} EqualOp;

} // namespace op::equal::cuda

#endif // __EQUAL_CUDA_H__
