#ifndef __BATCH_NORM_BACKWARD_KERNEL_CUH__
#define __BATCH_NORM_BACKWARD_KERNEL_CUH__
//  ------------------------------- start: perform operator on CUDA --------------------------------
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void batchNormBackwardKernel(
    Tdata * grad_input,
    Tdata * grad_weight,
    Tdata * grad_bias,
    const Tdata * input,
    const Tdata * grad_output,
    const Tdata * weight,
    const Tdata * running_mean,
    const Tdata * running_var,
    size_t batch_size,
    size_t channel_size,
    size_t dim_size,
    ptrdiff_t grad_weight_stride,
    ptrdiff_t grad_bias_stride,
    ptrdiff_t weight_stride,
    ptrdiff_t running_mean_stride,
    ptrdiff_t running_var_stride   
) {
    size_t c = blockIdx.x;
    Tcompute dbias = 0., dweight = 0., dvar = 0.;
    Tcompute std = sqrtf(Tcompute(*(running_var + c * running_var_stride)));
    Tcompute mean = Tcompute(*(running_mean + c * running_mean_stride));
    Tcompute wt = Tcompute(*(weight + c * weight_stride));
    for (size_t b = 0; b < batch_size; b++)
    {
        for (size_t d = 0; d < dim_size; d++)
        {
            size_t index = b * (channel_size * dim_size) + c * dim_size + d;
            dbias += Tcompute(*(grad_output + index));
            dweight += \
                Tcompute(grad_output[index]) * \
                (
                    Tcompute(input[index]) - mean
                );
            dvar += Tcompute(grad_output[index]) * \
            (
                Tcompute(input[index]) - mean
            ) * (-0.5) / (std * std * std); 
        }
    }
    *(grad_bias + c * grad_bias_stride) = Tdata(dbias);
    *(grad_weight + c * grad_weight_stride) = dweight / std;
    Tcompute dmu1 = -dbias * wt / std;
    for (size_t b = 0; b < batch_size; b++)
    {
        for (size_t d = 0; d < dim_size; d++)
        {
            size_t index = b * (channel_size * dim_size) + c * dim_size + d;
            float dx2 = 2 * dvar * wt * (Tcompute(input[index]) - mean);
            grad_input[index] = Tcompute(grad_output[index]) * wt / std + (dx2 + dmu1) / (batch_size * dim_size);
        }
    }
}
//  -------------------------------- end: perform operator on CUDA ---------------------------------

#endif // __BATCH_NORM_BACKWARD_KERNEL_CUH__
