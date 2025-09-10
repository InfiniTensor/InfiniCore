#ifndef GELU_CUDA_H
#define GELU_CUDA_H

namespace op::gelu::cuda {
typedef struct GeluOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            // For half2, process both elements in float precision
            const float sqrt_2_over_pi = 0.7978845608028654f;
            const float alpha = 0.044715f;
            const float half_val = 0.5f;
            const float one = 1.0f;

            // Convert half2 to float2 for calculations
            float2 x_f = __half22float2(x);

            // Calculate for x component
            float x_cubed_x = x_f.x * x_f.x * x_f.x;
            float inner_x = sqrt_2_over_pi * (x_f.x + alpha * x_cubed_x);
            float tanh_val_x = tanhf(inner_x);
            float result_x = half_val * x_f.x * (one + tanh_val_x);

            // Calculate for y component
            float x_cubed_y = x_f.y * x_f.y * x_f.y;
            float inner_y = sqrt_2_over_pi * (x_f.y + alpha * x_cubed_y);
            float tanh_val_y = tanhf(inner_y);
            float result_y = half_val * x_f.y * (one + tanh_val_y);

            // Convert back to half2
            float2 result_f = make_float2(result_x, result_y);
            return __float22half2_rn(result_f);
        } else if constexpr (std::is_same_v<T, half>) {
            const float sqrt_2_over_pi = 0.7978845608028654f;
            const float alpha = 0.044715f;
            const float half_val = 0.5f;
            const float one = 1.0f;

            // Convert to float for all calculations
            float x_f = __half2float(x);
            float x_cubed = x_f * x_f * x_f;
            float inner = sqrt_2_over_pi * (x_f + alpha * x_cubed);
            float tanh_val = tanhf(inner);
            float result = half_val * x_f * (one + tanh_val);

            return __float2half(result);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            const float sqrt_2_over_pi = 0.7978845608028654f;
            const float alpha = 0.044715f;
            const float half_val = 0.5f;
            const float one = 1.0f;

            // Convert to float for all calculations
            float x_f = __bfloat162float(x);
            float x_cubed = x_f * x_f * x_f;
            float inner = sqrt_2_over_pi * (x_f + alpha * x_cubed);
            float tanh_val = tanhf(inner);
            float result = half_val * x_f * (one + tanh_val);

            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, float>) {
            const float sqrt_2_over_pi = 0.7978845608028654f;
            const float alpha = 0.044715f;
            const float half_val = 0.5f;
            const float one = 1.0f;

            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + alpha * x_cubed);
            float tanh_val = tanhf(inner);

            return half_val * x * (one + tanh_val);
        } else if constexpr (std::is_same_v<T, double>) {
            const double sqrt_2_over_pi = 0.7978845608028654;
            const double alpha = 0.044715;
            const double half_val = 0.5;
            const double one = 1.0;

            double x_cubed = x * x * x;
            double inner = sqrt_2_over_pi * (x + alpha * x_cubed);
            double tanh_val = tanh(inner);

            return half_val * x * (one + tanh_val);
        } else {
            // Fallback for other types
            return x; // This should not happen with proper type checking
        }
    }
} GeluOp;
} // namespace op::gelu::cuda

#endif // GELU_CUDA_H