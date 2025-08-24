#ifndef __INFINIOP_LAYER_NORM_BACKWARD_API_H__
#define __INFINIOP_LAYER_NORM_BACKWARD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLayerNormBackwardDescriptor_t;

__C __export infiniStatus_t infiniopCreateLayerNormBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopLayerNormBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t grad_weight_desc,
    infiniopTensorDescriptor_t grad_bias_desc,
    infiniopTensorDescriptor_t grad_output_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    infiniopTensorDescriptor_t input_std_deviation_desc
);

__C __export infiniStatus_t infiniopGetLayerNormBackwardWorkspaceSize(infiniopLayerNormBackwardDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLayerNormBackward(
    infiniopLayerNormBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void * grad_input,
    void * grad_weight,
    void * grad_bias,
    const void * grad_output,
    const void * weight,
    const void * input_standardization,
    const void * input_std_deviation,
    void *stream
);

__C __export infiniStatus_t infiniopDestroyLayerNormBackwardDescriptor(infiniopLayerNormBackwardDescriptor_t desc);

#endif
