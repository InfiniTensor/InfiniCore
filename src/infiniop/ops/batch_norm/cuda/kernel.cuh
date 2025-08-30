#ifndef __BATCH_NORM_KERNEL_CUH__
#define __BATCH_NORM_KERNEL_CUH__

#include <cub/block/block_reduce.cuh>
#include "../../../reduce/cuda/reduce.cuh"

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void batchNormKernel(
    Tdata * output,
    Tdata * running_mean,
    Tdata * running_var,
    const Tdata * input,
    const Tdata * weight,
    const Tdata * bias,

    size_t batch_size,
    size_t channel_size,
    size_t dim_size,
    ptrdiff_t running_mean_stride,
    ptrdiff_t running_var_stride,
    ptrdiff_t weight_stride,
    ptrdiff_t bias_stride,
    float momentum,
    float eps    
) {
    auto output_ptr = output + dim_size * blockIdx.x;
    auto input_ptr = input + dim_size * blockIdx.x;
    
    auto running_mean_ptr = running_mean + running_mean_stride * blockIdx.x;
    auto running_var_ptr = running_var + running_var_stride * blockIdx.x;
    auto weight_ptr = weight + weight_stride * blockIdx.x;
    auto bias_ptr = bias + bias_stride * blockIdx.x;

    Tcompute sum_squared = 0., sum = 0.;
    for(size_t b = 0; b < batch_size; b++)
    {
        sum += op::common_cuda::reduce_op::sum<BLOCK_SIZE, Tdata, Tcompute>(
            input_ptr + b * (channel_size * dim_size), dim_size
        );
        sum_squared += op::common_cuda::reduce_op::sumSquared<BLOCK_SIZE, Tdata, Tcompute>(
            input_ptr + b * (channel_size * dim_size), dim_size
        );
    }
    
    __shared__ Tcompute E, var_biased;
    if (threadIdx.x == 0) {
        E = sum / Tcompute(batch_size * dim_size);
        var_biased = sum_squared / Tcompute(batch_size * dim_size) - E * E;
        Tcompute var_unbiased = var_biased * Tcompute(batch_size * dim_size) / Tcompute(batch_size * dim_size - 1);
        *running_mean_ptr = Tcompute(1 - momentum) * Tcompute(*running_mean_ptr) + Tcompute(momentum) * E;
        *running_var_ptr = Tcompute(1 - momentum) * Tcompute(*running_var_ptr) + Tcompute(momentum) * var_unbiased;
    }
    __syncthreads();

    for (size_t n = threadIdx.x; n < batch_size * dim_size; n += BLOCK_SIZE)
    {
        size_t b = n / dim_size, d = n % dim_size;
        *(output_ptr + b * channel_size * dim_size + d) = (
                Tcompute(*(input_ptr + b * channel_size * dim_size + d)) - E
            ) / sqrtf(float(var_biased + Tcompute(eps))) * Tcompute(*weight_ptr) + Tcompute(*bias_ptr);
    }
}

#endif // __BATCH_NORM_KERNEL_CUH__