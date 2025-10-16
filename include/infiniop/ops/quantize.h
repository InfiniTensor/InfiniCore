#ifndef __INFINIOP_QUANTIZE_API_H__
#define __INFINIOP_QUANTIZE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopQuantizeDescriptor_t;

__C __export infiniStatus_t infiniopCreateQuantizeDescriptor(
    infiniopHandle_t handle, infiniopQuantizeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t output_q_desc,
    infiniopTensorDescriptor_t output_s_desc);

__C __export infiniStatus_t infiniopGetQuantizeWorkspaceSize(
    infiniopQuantizeDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopQuantize(
    infiniopQuantizeDescriptor_t desc, void *workspace, size_t workspace_size,
    void *input, void *output_q, void *output_s, int group_size, double eps,
    double min_8bit, double max_8bit, bool scale_ue8m0, void *stream);

__C __export infiniStatus_t
infiniopDestroyQuantizeDescriptor(infiniopQuantizeDescriptor_t desc);
#endif