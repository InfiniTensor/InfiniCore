#pragma once

#include "../operator_descriptor.h"
#include <cstdint>

typedef struct InfiniopDescriptor *infiniopAminmaxDescriptor_t;

__C __export infiniStatus_t infiniopCreateAminmaxDescriptor(
    infiniopHandle_t handle,
    infiniopAminmaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t min_output_desc,
    infiniopTensorDescriptor_t max_output_desc,
    infiniopTensorDescriptor_t input_desc,
    int64_t dim,
    int32_t keepdim,
    int32_t has_dim);

__C __export infiniStatus_t
infiniopGetAminmaxWorkspaceSize(infiniopAminmaxDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopAminmax(
    infiniopAminmaxDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *min_output,
    void *max_output,
    const void *input,
    void *stream);

__C __export infiniStatus_t
infiniopDestroyAminmaxDescriptor(infiniopAminmaxDescriptor_t desc);
