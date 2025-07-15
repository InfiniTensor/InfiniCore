#ifndef __SWIGLU_MUSA_H__
#define __SWIGLU_MUSA_H__

#include "../../../elementwise/musa/elementwise_musa.h"
#include <musa_fp16.h>

namespace op::swiglu::musa {
typedef struct SwiGLUOp {
private:
    template <typename T>
    __device__ __forceinline__ T sigmoid(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            float2 x_float2 = make_float2(__half2float(x.x), __half2float(x.y));
            float2 result_float2 = make_float2(1.0f / (1.0f + std::exp(-x_float2.x)),
                                               1.0f / (1.0f + std::exp(-x_float2.y)));
            return make_half2(__float2half(result_float2.x),
                              __float2half(result_float2.y));
        } else if constexpr (std::is_same_v<T, half>) {
            float x_float = __half2float(x);
            float result_float = 1.0f / (1.0f + std::exp(-x_float));
            return __float2half(result_float);
        } else if constexpr (std::is_same_v<T, float>) {
            return 1.0f / (1.0f + std::exp(-x));
        } else {
            return 1.0 / (1.0 + std::exp(-x));
        }
    }

public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &up, const T &gate) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __hmul2(__hmul2(gate, sigmoid(gate)), up);
        } else if constexpr (std::is_same_v<T, half>) {
            return __hmul(__hmul(gate, sigmoid(gate)), up);
        } else if constexpr (std::is_same_v<T, float>) {
            return __fmul_rn(__fmul_rn(gate, sigmoid(gate)), up);
        } else {
            return gate * sigmoid(gate) * up;
        }
    }
} SwiGLUOp;
} // namespace op::swiglu::musa

#endif // __SWIGLU_MUSA_H__
