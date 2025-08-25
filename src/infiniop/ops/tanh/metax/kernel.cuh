#ifndef __TANH_METAX_H__
#define __TANH_METAX_H__

namespace op::tanh::metax {

typedef struct TanhOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half>) {
            float fx = __half2float(x);
            return __float2half(tanhf(fx));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float fx = __bfloat162float(x);
            return __float2bfloat16(tanhf(fx));
        } else if constexpr (std::is_same_v<T, float>) {
            return tanhf(x);
        } else {
            return ::tanh(x);
        }
    }
} TanhOp;

} // namespace op::tanh::metax

#endif // __TANH_METAX_H__