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
        
        if constexpr (std::is_same_v<T, half2>) {
            
            return __heq2(a, b);
        } 
        
        else if constexpr (std::is_same_v<T, half>) {
            
            return static_cast<T>(__heq(a, b));
        }
        
        else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            
             return static_cast<T>(a == b);
        }
        
        else {
            
            
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

} 

#endif 
