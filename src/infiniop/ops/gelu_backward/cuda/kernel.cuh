#ifndef __GELU_BACKWARD_CUDA_H__
#define __GELU_BACKWARD_CUDA_H__

namespace op::gelu_backward::cuda {
typedef struct GeluBackwardOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &input, const T &grad_output) const {
        if constexpr (std::is_same_v<T, half>) {
            return gelu_backward_half(input, grad_output);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return gelu_backward_bfloat16(input, grad_output);
        } else if constexpr (std::is_same_v<T, float>) {
            return gelu_backward_float(input, grad_output);
        } else if constexpr (std::is_same_v<T, double>) {
            return gelu_backward_double(input, grad_output);
        } else {
            return input + grad_output; // fallback
        }
    }

private:
    __device__ __forceinline__ half gelu_backward_half(const half &input, const half &grad_output) const {
        const half sqrt_2_over_pi = __float2half(0.7978845608028654f);
        const half coeff = __float2half(0.044715f);
        const half half_val = __float2half(0.5f);
        const half one = __float2half(1.0f);
        const half three = __float2half(3.0f);

        half x = input;
        half x_squared = __hmul(x, x);
        half x_cubed = __hmul(x_squared, x);
        half tanh_arg = __hmul(sqrt_2_over_pi, __hadd(x, __hmul(coeff, x_cubed)));
        half tanh_val = htanh(tanh_arg);

        half sech_squared = __hsub(one, __hmul(tanh_val, tanh_val));
        half arg_derivative = __hmul(sqrt_2_over_pi, __hadd(one, __hmul(__hmul(three, coeff), x_squared)));
        half gelu_derivative = __hadd(__hmul(half_val, __hadd(one, tanh_val)),
                                      __hmul(__hmul(x, half_val), __hmul(sech_squared, arg_derivative)));

        return __hmul(grad_output, gelu_derivative);
    }

    __device__ __forceinline__ cuda_bfloat16 gelu_backward_bfloat16(const cuda_bfloat16 &input, const cuda_bfloat16 &grad_output) const {
        return __float2bfloat16(gelu_backward_float(__bfloat162float(input), __bfloat162float(grad_output)));
    }

    __device__ __forceinline__ float gelu_backward_float(const float &input, const float &grad_output) const {
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        constexpr float coeff = 0.044715f;
        constexpr float half_val = 0.5f;
        constexpr float one = 1.0f;
        constexpr float three = 3.0f;

        float x = input;
        float x_squared = x * x;
        float x_cubed = x_squared * x;
        float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_val = tanhf(tanh_arg);

        float sech_squared = one - tanh_val * tanh_val;
        float arg_derivative = sqrt_2_over_pi * (one + three * coeff * x_squared);
        float gelu_derivative = half_val * (one + tanh_val) + x * half_val * sech_squared * arg_derivative;

        return grad_output * gelu_derivative;
    }

    __device__ __forceinline__ double gelu_backward_double(const double &input, const double &grad_output) const {
        constexpr double sqrt_2_over_pi = 0.7978845608028654;
        constexpr double coeff = 0.044715;
        constexpr double half_val = 0.5;
        constexpr double one = 1.0;
        constexpr double three = 3.0;

        double x = input;
        double x_squared = x * x;
        double x_cubed = x_squared * x;
        double tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        double tanh_val = tanh(tanh_arg);

        double sech_squared = one - tanh_val * tanh_val;
        double arg_derivative = sqrt_2_over_pi * (one + three * coeff * x_squared);
        double gelu_derivative = half_val * (one + tanh_val) + x * half_val * sech_squared * arg_derivative;

        return grad_output * gelu_derivative;
    }

    // Helper function for half precision tanh (assuming it exists or can be approximated)
    __device__ __forceinline__ half htanh(const half &x) const {
        return __float2half(tanhf(__half2float(x)));
    }

    // Helper function for bfloat16 precision tanh (assuming it exists or can be approximated)
    __device__ __forceinline__ cuda_bfloat16 htanh(const cuda_bfloat16 &x) const {
        return __float2bfloat16(tanhf(__bfloat162float(x)));
    }
} GeluBackwardOp;
} // namespace op::gelu_backward::cuda

#endif // __GELU_BACKWARD_CUDA_H__