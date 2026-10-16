#ifndef __INFINIOP_MOE_WEIGHTED_SUM_API_H__
#define __INFINIOP_MOE_WEIGHTED_SUM_API_H__

#include "../operator_descriptor.h"

__INFINI_C __export infiniStatus_t infiniopMoeWeightedSum(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t residual_desc,
    void *output,
    const void *input,
    const void *topk_weights,
    const void *residual,
    double routed_scale,
    double residual_scale,
    void *stream);

#endif
