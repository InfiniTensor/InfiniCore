#ifndef __LAYER_NORM_BACKWARD_KERNEL_CUH__
#define __LAYER_NORM_BACKWARD_KERNEL_CUH__

#include "../layer_norm_backward.h"
#include "../../../devices/metax/metax_handle.h"
#include "infinicore.h"

namespace op::layer_norm_backward::metax {

// Layer Norm Backward kernel for computing gradients
template <typename GradInputType, typename AccType, typename WeightType>
infiniStatus_t launchLayerNormBackwardKernel(
    void *grad_input_data,
    void *grad_weight_data,
    void *grad_bias_data,
    const void *grad_output_data,
    const void *input_data,
    const void *weight_data,
    const void *input_std_deviation_data,
    const void *input_standardization_data,
    const LayerNormBackwardInfo &info,
    hcStream_t stream);

// Sum up gradients for weight kernel
template <typename AccType, typename WeightType>
infiniStatus_t launchSumUpGradWKernel(
    void *grad_weight_data,
    void *grad_bias_data,
    const void *workspace,
    const LayerNormBackwardInfo &info,
    hcStream_t stream);

} // namespace op::layer_norm_backward::metax

#endif // __LAYER_NORM_BACKWARD_KERNEL_CUH__