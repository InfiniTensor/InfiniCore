#ifndef __SIGMOID_BACKWARD_CUDA_H__
#define __SIGMOID_BACKWARD_CUDA_H__


namespace op::sigmoid_backward::cuda {
    typedef struct SigmoidBackwardOp {
    public:
        static constexpr size_t num_inputs = 2;
        template <typename T>
        __device__ __forceinline__ T operator()(const T &grad_output, const T &input) const {
            if constexpr (std::is_same_v<T, half>) {
                float x = __half2float(input);
                float grad = __half2float(grad_output);
                float sig = 1.0f / (1.0f + __expf(-x));
                float dy_dx = sig * (1.0f - sig);
                return __float2half(grad * dy_dx);
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                float x = __bfloat162float(input);
                float grad = __bfloat162float(grad_output);
                float sig;
                if (x >= 0.f) {
                    float z = ::expf(-x);
                    sig = 1.f / (1.f + z);
                } else {
                    float z = ::expf(x);
                    sig = z / (1.f + z);
                }
                float dy_dx = sig * (1.f - sig);
                return __float2bfloat16(grad * dy_dx);
            } else if constexpr (std::is_same_v<T, float>) {
                float sig = 1.0f / (1.0f + __expf(-input));
                return grad_output * sig * (1.0f - sig);
            } else if constexpr (std::is_same_v<T, double>) {
                double sig = 1.0 / (1.0 + ::exp(-input));
                return grad_output * sig * (1.0 - sig);
            } else {
                // fallback to double for other types
                double x = static_cast<double>(input);
                double grad = static_cast<double>(grad_output);
                double sig = 1.0 / (1.0 + ::exp(-x));
                double dy_dx = sig * (1.0 - sig);
                return static_cast<T>(grad * dy_dx);
            }
        }
    } SigmoidBackwardOp;
}

#endif

