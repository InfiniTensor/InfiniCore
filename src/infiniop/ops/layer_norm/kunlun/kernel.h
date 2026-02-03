#ifndef __LAYER_NORM_KUNLUN_KERNEL_H__
#define __LAYER_NORM_KUNLUN_KERNEL_H__

#include "../../../devices/kunlun/kunlun_kernel_common.h"
#include "../../../reduce/kunlun/reduce_kunlun.h"

using namespace device::kunlun::kernel;

// Calculate norm in BLOCK_SIZE cores in one cluster. Useful for long normalized_size
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void layerNormCluster(
    __shared_ptr__ Tdata *y,
    __shared_ptr__ Tdata *output_standardized,
    __shared_ptr__ Tdata *output_std_deviation,
    __shared_ptr__ const Tdata *input,
    __shared_ptr__ const Tdata *weight,
    __shared_ptr__ const Tdata *bias,
    float eps,
    int32_t normalized_size,
    bool bias_exist) {

    // Block reduce sum of x^2
    Tcompute mean = op::common_kunlun::reduce_op::
                        sum<BLOCK_SIZE, Tdata, Tcompute>(input, normalized_size)
                  / normalized_size;
    Tcompute sum_squared = op::common_kunlun::reduce_op::
        sumSquared<BLOCK_SIZE, Tdata, Tcompute>(input, normalized_size);
    Tcompute var = sum_squared / normalized_size - mean * mean;
    // Compute rsqrt variance + epsilon
    Tcompute rstd = Tcompute(1.0f) / sqrt(var + Tcompute(eps));

    // Write to output_std_deviation
    if (core_id() == 0) {
        *output_std_deviation = static_cast<Tdata>(rstd);
    }
    sync_cluster();

    for (int32_t i = core_id(); i < normalized_size; i += BLOCK_SIZE) {
        Tcompute x_standard = (Tcompute(input[i]) - mean) * rstd;
        output_standardized[i] = static_cast<Tdata>(x_standard);
        y[i] = static_cast<Tdata>(x_standard * Tcompute(weight[i]) + (bias_exist ? Tcompute(bias[i]) : Tcompute(0)));
    }
    sync_cluster();
}

// Calculate norm in single core. Useful for short normalized_size
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void layerNormBlock(
    __local__ Tdata *output,
    __local__ Tdata *output_standardization,
    __local__ Tdata *output_rstd_deviation,
    __local__ const Tdata *input,
    __shared_ptr__ const Tdata *weight,
    __shared_ptr__ const Tdata *bias,
    float eps,
    int32_t normalized_size,
    bool bias_exist) {

    // Block reduce sum of x^2
    Tcompute mean = op::common_kunlun::reduce_op::blockSum<Tdata, Tcompute>(input, normalized_size)
                  / normalized_size;
    Tcompute sum_squared = op::common_kunlun::reduce_op::blockSumSquared<Tdata, Tcompute>(input, normalized_size);
    Tcompute var = sum_squared / normalized_size - mean * mean;
    // Compute rsqrt variance + epsilon
    Tcompute rstd = Tcompute(1.0f) / sqrt(var + Tcompute(eps));
    // Write to output_rstd_deviation
    *output_rstd_deviation = static_cast<Tdata>(rstd);

    for (int32_t i = 0; i < normalized_size; i += 1) {
        Tcompute x_standard = (Tcompute(input[i]) - mean) * rstd;
        output_standardization[i] = static_cast<Tdata>(x_standard);
        output[i] = static_cast<Tdata>(x_standard * Tcompute(weight[i]) + (bias_exist ? Tcompute(bias[i]) : Tcompute(0)));
    }
    mfence();
}

#endif
