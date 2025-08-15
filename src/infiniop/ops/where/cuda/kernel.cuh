#ifndef __WHERE_CUDA_H__
#define __WHERE_CUDA_H__

namespace op::where::cuda {

typedef struct WhereOp {
public:
    static constexpr size_t num_inputs = 3;
    
    // Template version for mixed data types
    template <typename Tout, typename Tcond, typename Ta, typename Tb>
    __device__ __forceinline__ Tout operator()(const Tcond &condition, const Ta &a, const Tb &b) const {
        return condition ? static_cast<Tout>(a) : static_cast<Tout>(b);
    }
    
    template <typename T>
    __device__ __forceinline__ T operator()(const bool &condition, const T &a, const T &b) const {
        return condition ? a : b;
    }
    
    // 为half2类型特化
    __device__ __forceinline__ half2 operator()(const bool &condition, const half2 &a, const half2 &b) const {
        return condition ? a : b;
    }
    
    // 为half类型特化
    __device__ __forceinline__ half operator()(const bool &condition, const half &a, const half &b) const {
        return condition ? a : b;
    }
    
    // 为cuda_bfloat16类型特化
    __device__ __forceinline__ cuda_bfloat16 operator()(const bool &condition, const cuda_bfloat16 &a, const cuda_bfloat16 &b) const {
        return condition ? a : b;
    }
    
    // 为float类型特化
    __device__ __forceinline__ float operator()(const bool &condition, const float &a, const float &b) const {
        return condition ? a : b;
    }
    
    // 为double类型特化
    __device__ __forceinline__ double operator()(const bool &condition, const double &a, const double &b) const {
        return condition ? a : b;
    }
    
    // 为int8_t类型特化
    __device__ __forceinline__ int8_t operator()(const bool &condition, const int8_t &a, const int8_t &b) const {
        return condition ? a : b;
    }
    
    // 为int16_t类型特化
    __device__ __forceinline__ int16_t operator()(const bool &condition, const int16_t &a, const int16_t &b) const {
        return condition ? a : b;
    }
    
    // 为int32_t类型特化
    __device__ __forceinline__ int32_t operator()(const bool &condition, const int32_t &a, const int32_t &b) const {
        return condition ? a : b;
    }
    
    // 为int64_t类型特化
    __device__ __forceinline__ int64_t operator()(const bool &condition, const int64_t &a, const int64_t &b) const {
        return condition ? a : b;
    }
    
    // 为uint8_t类型特化
    __device__ __forceinline__ uint8_t operator()(const bool &condition, const uint8_t &a, const uint8_t &b) const {
        return condition ? a : b;
    }
    
    // 为uint16_t类型特化
    __device__ __forceinline__ uint16_t operator()(const bool &condition, const uint16_t &a, const uint16_t &b) const {
        return condition ? a : b;
    }
    
    // 为uint32_t类型特化
    __device__ __forceinline__ uint32_t operator()(const bool &condition, const uint32_t &a, const uint32_t &b) const {
        return condition ? a : b;
    }
    
    // 为uint64_t类型特化
    __device__ __forceinline__ uint64_t operator()(const bool &condition, const uint64_t &a, const uint64_t &b) const {
        return condition ? a : b;
    }
} WhereOp;

// 高精度版本（与标准版本相同，因为where操作本身不涉及复杂计算）
typedef struct WhereOpHighPrecision {
public:
    static constexpr size_t num_inputs = 3;
    
    template <typename T>
    __device__ __forceinline__ T operator()(const bool &condition, const T &a, const T &b) const {
        return condition ? a : b;
    }
    
    // 为half2类型特化
    __device__ __forceinline__ half2 operator()(const bool &condition, const half2 &a, const half2 &b) const {
        return condition ? a : b;
    }
    
    // 为half类型特化
    __device__ __forceinline__ half operator()(const bool &condition, const half &a, const half &b) const {
        return condition ? a : b;
    }
    
    // 为cuda_bfloat16类型特化
    __device__ __forceinline__ cuda_bfloat16 operator()(const bool &condition, const cuda_bfloat16 &a, const cuda_bfloat16 &b) const {
        return condition ? a : b;
    }
    
    // 为float类型特化
    __device__ __forceinline__ float operator()(const bool &condition, const float &a, const float &b) const {
        return condition ? a : b;
    }
    
    // 为double类型特化
    __device__ __forceinline__ double operator()(const bool &condition, const double &a, const double &b) const {
        return condition ? a : b;
    }
    
    // 为int8_t类型特化
    __device__ __forceinline__ int8_t operator()(const bool &condition, const int8_t &a, const int8_t &b) const {
        return condition ? a : b;
    }
    
    // 为int16_t类型特化
    __device__ __forceinline__ int16_t operator()(const bool &condition, const int16_t &a, const int16_t &b) const {
        return condition ? a : b;
    }
    
    // 为int32_t类型特化
    __device__ __forceinline__ int32_t operator()(const bool &condition, const int32_t &a, const int32_t &b) const {
        return condition ? a : b;
    }
    
    // 为int64_t类型特化
    __device__ __forceinline__ int64_t operator()(const bool &condition, const int64_t &a, const int64_t &b) const {
        return condition ? a : b;
    }
    
    // 为uint8_t类型特化
    __device__ __forceinline__ uint8_t operator()(const bool &condition, const uint8_t &a, const uint8_t &b) const {
        return condition ? a : b;
    }
    
    // 为uint16_t类型特化
    __device__ __forceinline__ uint16_t operator()(const bool &condition, const uint16_t &a, const uint16_t &b) const {
        return condition ? a : b;
    }
    
    // 为uint32_t类型特化
    __device__ __forceinline__ uint32_t operator()(const bool &condition, const uint32_t &a, const uint32_t &b) const {
        return condition ? a : b;
    }
    
    // 为uint64_t类型特化
    __device__ __forceinline__ uint64_t operator()(const bool &condition, const uint64_t &a, const uint64_t &b) const {
        return condition ? a : b;
    }
} WhereOpHighPrecision;

} // namespace op::where::cuda

#endif // __WHERE_CUDA_H__