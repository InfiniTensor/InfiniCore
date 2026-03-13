#ifndef __MUL_CUDA_H__
#define __MUL_CUDA_H__

namespace op::mul::cuda {
typedef struct MulOp {
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half2> || std::is_same_v<T, cuda_bfloat162>) {
            return __hmul2(a, b);
        } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
            return __hmul(a, b);
        } else if constexpr (std::is_same_v<T, float>) {
            return __fmul_rn(a, b);
        } else {
            return a * b;
        }
    }
} MulOp;

/**
 * @brief 支持两个不同类型相乘的 Mul 算子
 *
 * 用于 elementwise 混合类型 kernel: Op{}.template operator()<Tout, TA, TB>(a, b)
 */
typedef struct MulOpv2 {
    static constexpr size_t num_inputs = 2;

    template <typename Tout, typename TA, typename TB>
    __device__ __forceinline__ Tout operator()(const TA &a, const TB &b) const {
        if constexpr (std::is_same_v<TA, TB>) {
            return mul_same<Tout>(a, b);
        } else {
            return mul_mixed<Tout>(a, b);
        }
    }

private:
    template <typename Tout, typename T>
    static __device__ __forceinline__ Tout mul_same(const T &a, const T &b) {
        if constexpr (std::is_same_v<T, half2> || std::is_same_v<T, cuda_bfloat162>) {
            return static_cast<Tout>(__hmul2(a, b));
        } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
            return static_cast<Tout>(__hmul(a, b));
        } else if constexpr (std::is_same_v<T, float>) {
            return static_cast<Tout>(__fmul_rn(a, b));
        } else {
            return static_cast<Tout>(a * b);
        }
    }

    template <typename Tout, typename TA, typename TB>
    static __device__ __forceinline__ Tout mul_mixed(const TA &a, const TB &b) {
        // 混合类型：转为 float 或 double 计算后转回 Tout
        float fa = to_float(a);
        float fb = to_float(b);
        return to_tout<Tout>(__fmul_rn(fa, fb));
    }

    template <typename T>
    static __device__ __forceinline__ float to_float(const T &x) {
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __half>) {
            return __half2float(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __bfloat162float(x);
        } else {
            return static_cast<float>(x);
        }
    }

    template <typename Tout, typename Tin>
    static __device__ __forceinline__ Tout to_tout(Tin val) {
        if constexpr (std::is_same_v<Tout, half> || std::is_same_v<Tout, __half>) {
            return __float2half_rn(static_cast<float>(val));
        } else if constexpr (std::is_same_v<Tout, cuda_bfloat16>) {
            return __float2bfloat16_rn(static_cast<float>(val));
        } else if constexpr (std::is_same_v<Tout, float>) {
            return static_cast<float>(val);
        } else if constexpr (std::is_same_v<Tout, double>) {
            return static_cast<double>(val);
        } else {
            return static_cast<Tout>(val);
        }
    }
} MulOpv2;

} // namespace op::mul::cuda

#endif // __MUL_CUDA_H__
