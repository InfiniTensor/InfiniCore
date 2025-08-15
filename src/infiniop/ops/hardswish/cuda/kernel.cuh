#ifndef __HARDSWISH_CUDA_H__
#define __HARDSWISH_CUDA_H__

namespace op::hardswish::cuda {

// HardSwish函数的CUDA实现
// HardSwish(x) = x * ReLU6(x + 3) / 6
// 其中 ReLU6(x) = min(max(x, 0), 6)

// 快速HardSwish实现
template<typename T>
__device__ __forceinline__ T fast_hardswish(T x) {
    float fx;
    if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        fx = __bfloat162float(x);
    } else {
        fx = static_cast<float>(x);
    }
    
    // 计算 x + 3
    float x_plus_3 = fx + 3.0f;
    
    // 计算 ReLU6(x + 3) = min(max(x + 3, 0), 6)
    float relu6_result = fminf(fmaxf(x_plus_3, 0.0f), 6.0f);
    
    // 计算 x * ReLU6(x + 3) / 6
    float result = fx * relu6_result / 6.0f;
    
    if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        return __float2bfloat16(result);
    } else {
        return static_cast<T>(result);
    }
}

// 高精度HardSwish实现
template<typename T>
__device__ __forceinline__ T precise_hardswish(T x) {
    if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        float x_float = __bfloat162float(x);
        double x_double = static_cast<double>(x_float);
        
        // 使用double精度计算
        double x_plus_3 = x_double + 3.0;
        double relu6_result = fmin(fmax(x_plus_3, 0.0), 6.0);
        double result = x_double * relu6_result / 6.0;
        
        return __float2bfloat16(static_cast<float>(result));
    } else if constexpr (std::is_same_v<T, float>) {
        float x_plus_3 = x + 3.0f;
        float relu6_result = fminf(fmaxf(x_plus_3, 0.0f), 6.0f);
        return x * relu6_result / 6.0f;
    } else {
        // 对于half类型，直接使用float计算然后转换
        float fx = static_cast<float>(x);
        float x_plus_3 = fx + 3.0f;
        float relu6_result = fminf(fmaxf(x_plus_3, 0.0f), 6.0f);
        float result = fx * relu6_result / 6.0f;
        return static_cast<T>(result);
    }
}

// HardSwish算子结构体
typedef struct HardSwishOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            // 对于half2，分别处理两个half值
            half x1 = __low2half(x);
            half x2 = __high2half(x);
            half y1 = fast_hardswish(x1);
            half y2 = fast_hardswish(x2);
            return __halves2half2(y1, y2);
        } else if constexpr (std::is_same_v<T, half>) {
            return fast_hardswish(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return fast_hardswish(x);
        } else if constexpr (std::is_same_v<T, float>) {
            return fast_hardswish(x);
        } else {
            return fast_hardswish(x);
        }
    }
} HardSwishOp;

// 高精度版本的HardSwish算子
typedef struct HardSwishOpHighPrecision {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            // 对于half2，分别处理两个half值
            half x1 = __low2half(x);
            half x2 = __high2half(x);
            half y1 = precise_hardswish(x1);
            half y2 = precise_hardswish(x2);
            return __halves2half2(y1, y2);
        } else if constexpr (std::is_same_v<T, half>) {
            return precise_hardswish(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return precise_hardswish(x);
        } else if constexpr (std::is_same_v<T, float>) {
            return precise_hardswish(x);
        } else {
            return precise_hardswish(x);
        }
    }
} HardSwishOpHighPrecision;

} // namespace op::hardswish::cuda

#endif // __HARDSWISH_CUDA_H__