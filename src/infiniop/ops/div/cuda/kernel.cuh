#ifndef __DIV_CUDA_H__
#define __DIV_CUDA_H__

// #include "../../../devices/nvidia/nvidia_kernel_common.cuh"
namespace op::div::cuda {
typedef struct DivOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half>) {
            return __hdiv(a, b);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __hdiv(a, b);
        } else if constexpr (std::is_same_v<T, float>) {
            return fdividef(a, b);
        } else if constexpr (std::is_same_v<T, double>) {
            return fdivide(a, b);
        } else {
            return a / b;
        }
    }
} DivOp;
typedef struct DivOpTrunc {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half>) {
            return htrunc(__hdiv(a, b));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return htrunc(__hdiv(a, b));
        } else if constexpr (std::is_same_v<T, float>) {
            return truncf(fdividef(a, b));
        } else if constexpr (std::is_same_v<T, double>) {
            return trunc(fdivide(a, b));
        } else {
            return a / b;
        }
    }
} DivOpTrunc;
typedef struct DivOpFloor {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half>) {
            float fa = __half2float(a);
            float fb = __half2float(b);
            float res = floorf(fdividef(fa, fb));
            return __float2half(res);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float fa = __bfloat162float(a);
            float fb = __bfloat162float(b);
            float res = floorf(fdividef(fa, fb));
            return __float2bfloat16(res);

        } else if constexpr (std::is_same_v<T, float>) {
            // return floorf(fdividef(a, b));
            return floorf(a / b);
        } else if constexpr (std::is_same_v<T, double>) {
            return floor(fdivide(a, b));
        } else {
            return a / b;
        }
    }
} DivOpFloor;
} // namespace op::div::cuda

#endif // __DIV_CUDA_H__
