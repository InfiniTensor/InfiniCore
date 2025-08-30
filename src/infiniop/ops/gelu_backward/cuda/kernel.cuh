#ifndef __GELU_BACKWARD_CUDA_H__
#define __GELU_BACKWARD_CUDA_H__

namespace op::gelu_backward::cuda {
typedef struct GeluBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        constexpr float alpha = 0.7978845608028654f;
        constexpr float beta = 0.044715f;
        constexpr float beta3 = 3.0f * beta;

        if constexpr (std::is_same_v<T, half2>) {
            // half2向量化优化
            float2 x_f = __half22float2(a);
            float2 grad_output_f = __half22float2(b);

            float2 u = {
                alpha * (x_f.x + beta * x_f.x * x_f.x * x_f.x),
                alpha * (x_f.y + beta * x_f.y * x_f.y * x_f.y)};
            // 分别计算 tanh 和 sech²
            float tanh_u_x = tanhf(u.x);
            float tanh_u_y = tanhf(u.y);
            float sech2_u_x = 1.0f - tanh_u_x * tanh_u_x;
            float sech2_u_y = 1.0f - tanh_u_y * tanh_u_y;
            // 分别计算导数分量
            float du_dx_x = alpha * (1.0f + beta3 * x_f.x * x_f.x);
            float du_dx_y = alpha * (1.0f + beta3 * x_f.y * x_f.y);
            float dy_dx_x = 0.5f * (1.0f + tanh_u_x) + 0.5f * x_f.x * sech2_u_x * du_dx_x;
            float dy_dx_y = 0.5f * (1.0f + tanh_u_y) + 0.5f * x_f.y * sech2_u_y * du_dx_y;

            float2 grad_input_f = {
                grad_output_f.x * dy_dx_x,
                grad_output_f.y * dy_dx_y};
            return __float22half2_rn(grad_input_f);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16精度
            float x_f = __bfloat162float(a);
            float grad_output_f = __bfloat162float(b);

            float u = alpha * (x_f + beta * x_f * x_f * x_f);
            float tanh_u = tanhf(u);
            float sech2_u = 1.0f - tanh_u * tanh_u;
            float du_dx = alpha * (1.0f + beta3 * x_f * x_f);
            float dy_dx = 0.5f * (1.0f + tanh_u) + 0.5f * x_f * sech2_u * du_dx;
            float ans = __fmul_rn(grad_output_f, dy_dx);

            return __float2bfloat16(ans);
        } else if constexpr (std::is_same_v<T, half>) {
            // FP16精度
            float x_f = __half2float(a);
            float grad_output_f = __half2float(b);

            float u = alpha * (x_f + beta * x_f * x_f * x_f);
            float tanh_u = tanhf(u);
            float sech2_u = 1.0f - tanh_u * tanh_u;
            float du_dx = alpha * (1.0f + beta3 * x_f * x_f);
            float dy_dx = 0.5f * (1.0f + tanh_u) + 0.5f * x_f * sech2_u * du_dx;
            float ans = __fmul_rn(grad_output_f, dy_dx);

            return __float2half(ans);
        } else if constexpr (std::is_same_v<T, float>) {
            // FP32精度
            float x = a;
            float u = alpha * (x + beta * x * x * x);
            float tanh_u = tanhf(u);
            float sech2_u = 1.0f - tanh_u * tanh_u;
            float du_dx = alpha * (1.0f + beta3 * x * x);
            float dy_dx = 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2_u * du_dx;
            return __fmul_rn(b, dy_dx);
        } else {
            // FP64精度或其他
            constexpr double alpha_d = 0.7978845608028654;
            constexpr double beta_d = 0.044715;
            double x = a;
            double u = alpha_d * (x + beta_d * x * x * x);
            double tanh_u = tanh(u);
            double sech2_u = 1.0 - tanh_u * tanh_u;
            double du_dx = alpha_d * (1.0 + 3.0 * beta_d * x * x);
            double dy_dx = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx;
            return static_cast<T>(b * dy_dx);
        }
    }
} GeluBackwardOp;
} // namespace op::gelu_backward::cuda

#endif // __GELU_BACKWARD_CUDA_H__
