#ifndef __LAYER_NORM_KERNEL_CUH__
#define __LAYER_NORM_KERNEL_CUH__
//  ------------------------------- start: perform operator on CUDA --------------------------------
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void layerNormKernel(
    Tdata * output,
    Tdata * input_standardization,
    Tdata * input_std_deviation,
    const Tdata * input,
    const Tdata * weight,
    const Tdata * bias,
    float eps,
    size_t normalized_size,
    const ptrdiff_t* output_strides,
    const ptrdiff_t* input_standardization_strides,
    const ptrdiff_t* input_std_deviation_strides,
    const ptrdiff_t* input_strides,
    ptrdiff_t weight_stride,
    ptrdiff_t bias_stride,
    bool bias_exist
) {
    size_t b0 = blockIdx.x, b1 = blockIdx.y;

    auto output_ptr = output + b0 * output_strides[0] + b1 * output_strides[1];
    auto input_ptr = input + b0 * input_strides[0] + b1 * input_strides[1];
    auto standard_ptr = input_standardization + b0 * input_standardization_strides[0] + b1 * input_standardization_strides[1];
    auto std_ptr = input_std_deviation + b0 * input_std_deviation_strides[0] + b1 * input_std_deviation_strides[1];
    Tcompute mean = op::common_cuda::reduce_op::sum<BLOCK_SIZE, Tdata, Tcompute>(
        input_ptr,
        normalized_size
    ) / normalized_size;
    Tcompute sum_squared = op::common_cuda::reduce_op::sumSquared<BLOCK_SIZE, Tdata, Tcompute>(
        input_ptr,
        normalized_size
    );

    Tcompute var = sum_squared / normalized_size - mean * mean;
    Tcompute std_deviation = sqrtf(var + Tcompute(eps));
    *std_ptr = std_deviation;

    for (size_t d = 0; d < normalized_size; d ++) {
        Tcompute x_standard = (Tcompute(input_ptr[d]) - mean) / std_deviation;
        standard_ptr[d] = x_standard;
        output_ptr[d] = x_standard * Tcompute(*(weight + d * weight_stride)) + (bias_exist ? Tcompute(*(bias + d * bias_stride)) : Tcompute(0));
    }
}
//  -------------------------------- end: perform operator on CUDA ---------------------------------

#endif // __LAYER_NORM_KERNEL_CUH__
