#ifndef __INFINIOP_AFFINE_GRID_API_H__
#define __INFINIOP_AFFINE_GRID_API_H__
#include <stdint.h>
#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAffineGridDescriptor_t;

__C __export infiniStatus_t infiniopCreateAffineGridDescriptor(infiniopHandle_t handle,
                                                               infiniopAffineGridDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t output_desc,
                                                               infiniopTensorDescriptor_t input_desc,
                                                               uint8_t align_corners);

__C __export infiniStatus_t infiniopGetAffineGridWorkspaceSize(infiniopAffineGridDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopAffineGrid(infiniopAffineGridDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *output,
                                               const void *input,
                                               void *stream);

__C __export infiniStatus_t infiniopDestroyAffineGridDescriptor(infiniopAffineGridDescriptor_t desc);

#endif // __INFINIOP_AFFINE_GRID_API_H__