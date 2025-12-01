#pragma once

#ifndef __INFINIOP_DIAGFLAT_API_H__
#define __INFINIOP_DIAGFLAT_API_H__

#include "../operator_descriptor.h"
#include <cstdint>

typedef struct InfiniopDescriptor *infiniopDiagflatDescriptor_t;

__C __export infiniStatus_t infiniopCreateDiagflatDescriptor(
    infiniopHandle_t handle,
    infiniopDiagflatDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input,
    int64_t offset);

__C __export infiniStatus_t
infiniopGetDiagflatWorkspaceSize(infiniopDiagflatDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopDiagflat(
    infiniopDiagflatDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

__C __export infiniStatus_t
infiniopDestroyDiagflatDescriptor(infiniopDiagflatDescriptor_t desc);

#endif // __INFINIOP_DIAGFLAT_API_H__