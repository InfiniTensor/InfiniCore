#ifndef __EXP_CUDA_H__
#define __EXP_CUDA_H__


namespace op::exp::cuda {
typedef struct ExpOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a) const {
        if constexpr (std::is_same_v<T, half>) {
            return hexp(a);  // 半精度
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __expf(__bfloat162float(a));  // 先转成float再exp
        } else if constexpr (std::is_same_v<T, float>) {
            return __expf(a);  // 快速 float 指数函数
        } else if constexpr (std::is_same_v<T, double>) {
            return ::exp(a);  // 双精度标准库
        } else {
            return ::exp(a); 
        }
    }
} ExpOp;
} // namespace op::exp::cuda

#endif // __EXP_CUDA_H__
