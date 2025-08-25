#ifndef __COS_METAX_H__
#define __COS_METAX_H__

namespace op::cos::metax {

typedef struct CosOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2cos(x);
        } else if constexpr (std::is_same_v<T, half>) {
            return hcos(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // 使用double作为中间计算类型以提高精度
            double x_double = static_cast<double>(__bfloat162float(x));
            double result = ::cos(x_double);
            return __float2bfloat16(static_cast<float>(result));
        } else if constexpr (std::is_same_v<T, float>) {
            return cosf(x);
        } else {
            return ::cos(x);
        }
    }
} CosOp;

} // namespace op::cos::metax

#endif // __COS_METAX_H__