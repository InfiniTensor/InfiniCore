#ifndef __LAYER_NORM_BACKWARD_KERNEL_CUH__
#define __LAYER_NORM_BACKWARD_KERNEL_CUH__
//  ------------------------------- start: perform operator on CUDA --------------------------------
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void layerNormBackwardStepOneKernel(
    Tdata * grad_input,
    Tdata * grad_weight,
    Tdata * grad_bias,
    const Tdata * grad_output,
    const Tdata * weight,
    const Tdata * input_standardization,
    const Tdata * input_std_deviation,
    size_t batch_size,
    size_t channel_size,
    size_t feature_size,
    ptrdiff_t grad_output_stride_b,
    ptrdiff_t grad_output_stride_c,
    ptrdiff_t input_standardization_stride_b,
    ptrdiff_t input_standardization_stride_c,
    ptrdiff_t input_std_deviation_stride_b,
    ptrdiff_t input_std_deviation_stride_c,
    ptrdiff_t grad_weight_stride,
    ptrdiff_t grad_bias_stride,
    bool bias
) {
    size_t feature_index = blockIdx.x;
    auto grad_output_ptr = grad_output + feature_index;
    auto input_standard_ptr = input_standardization + feature_index;
    Tcompute sum_dy = 0.;
    Tcompute sum_dy_norm_x = 0.;
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < channel_size; c ++) {
            Tcompute norm_x = Tcompute(*(input_standard_ptr + b * input_standardization_stride_b + c * input_standardization_stride_c));
            Tcompute dy = Tcompute(*(grad_output_ptr + b * grad_output_stride_b + c * grad_output_stride_c));
            sum_dy += dy;
            sum_dy_norm_x += dy * norm_x;
        }
    }
    *(grad_weight + feature_index * grad_weight_stride) = sum_dy_norm_x;
    if (bias)
        *(grad_bias + feature_index * grad_bias_stride) = sum_dy;
}


template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void layerNormBackwardStepTwoKernel(
    Tdata * grad_input,
    Tdata * grad_weight,
    Tdata * grad_bias,
    const Tdata * grad_output,
    const Tdata * weight,
    const Tdata * input_standardization,
    const Tdata * input_std_deviation,

    size_t batch_size,
    size_t channel_size,
    size_t feature_size,

    ptrdiff_t grad_input_stride_b,
    ptrdiff_t grad_input_stride_c,
    ptrdiff_t grad_output_stride_b,
    ptrdiff_t grad_output_stride_c,  
    ptrdiff_t weight_stride,
    ptrdiff_t input_standardization_stride_b,
    ptrdiff_t input_standardization_stride_c,
    ptrdiff_t input_std_deviation_stride_b,
    ptrdiff_t input_std_deviation_stride_c  
) {
    size_t b = blockIdx.x, c = blockIdx.y;
        
    Tcompute std = *(input_std_deviation + b * input_std_deviation_stride_b + c * input_std_deviation_stride_c);
    auto grad_output_ptr = grad_output + b * grad_output_stride_b + c * grad_output_stride_c;
    auto input_standard_ptr = input_standardization + b * input_standardization_stride_b + c * input_standardization_stride_c;
    auto grad_input_ptr = grad_input + b * grad_input_stride_b + c * grad_input_stride_c;

    __shared__ Tcompute sum_dy_w;
    __shared__ Tcompute sum_dy_w_norm_x;
    if (threadIdx.x == 0) {
        sum_dy_w = 0;
        sum_dy_w_norm_x = 0;
        for (size_t i = 0; i < feature_size; i ++) {
            Tcompute wt = *(weight + i * weight_stride);
            Tcompute dy_w = Tcompute(grad_output_ptr[i]) * wt;
            Tcompute norm_x = Tcompute(input_standard_ptr[i]);
            sum_dy_w += dy_w;
            sum_dy_w_norm_x += dy_w * norm_x;
        }
    }
    __syncthreads();

    for (size_t i = threadIdx.x; i < feature_size; i += BLOCK_SIZE) {
        Tcompute wt = *(weight + i * weight_stride);
        Tcompute dy = grad_output_ptr[i];
        Tcompute norm_x = Tcompute(input_standard_ptr[i]);
        grad_input_ptr[i] = wt * dy / std + (
            - sum_dy_w - norm_x * sum_dy_w_norm_x
        ) / (std * feature_size);
    }  
}
//  -------------------------------- end: perform operator on CUDA ---------------------------------

#endif // __LAYER_NORM_BACKWARD_KERNEL_CUH__
