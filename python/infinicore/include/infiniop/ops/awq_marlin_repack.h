#ifndef __INFINIOP_AWQ_MARLIN_REPACK_API_H__
#define __INFINIOP_AWQ_MARLIN_REPACK_API_H__

#include "../operator_descriptor.h"
#include <cstdint>

typedef struct InfiniopDescriptor *infiniopAwqMarlinRepackDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAwqMarlinRepackDescriptor(infiniopHandle_t handle,
                                                                           infiniopAwqMarlinRepackDescriptor_t *desc_ptr,
                                                                           infiniopTensorDescriptor_t output_desc,
                                                                           infiniopTensorDescriptor_t input_desc,
                                                                           int64_t num_bits,
                                                                           bool is_a_8bit);

__INFINI_C __export infiniStatus_t infiniopGetAwqMarlinRepackWorkspaceSize(infiniopAwqMarlinRepackDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAwqMarlinRepack(infiniopAwqMarlinRepackDescriptor_t desc,
                                                           void *workspace,
                                                           size_t workspace_size,
                                                           void *output,
                                                           const void *input,
                                                           void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAwqMarlinRepackDescriptor(infiniopAwqMarlinRepackDescriptor_t desc);

#endif
