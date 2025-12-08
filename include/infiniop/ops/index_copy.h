#ifndef __INFINIOP_INDEX_COPY_API_H__
#define __INFINIOP_INDEX_COPY_API_H__
#include <stdint.h>
#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopIndexCopyDescriptor_t;

__C __export infiniStatus_t infiniopCreateIndexCopyDescriptor(infiniopHandle_t handle,
                                                             infiniopIndexCopyDescriptor_t *desc_ptr,
                                                             infiniopTensorDescriptor_t output,
                                                             infiniopTensorDescriptor_t input,
                                                             int64_t dim,
                                                             infiniopTensorDescriptor_t index,
                                                             infiniopTensorDescriptor_t source);

__C __export infiniStatus_t infiniopGetIndexCopyWorkspaceSize(infiniopIndexCopyDescriptor_t desc, size_t *size);
__C __export infiniStatus_t infiniopIndexCopy(infiniopIndexCopyDescriptor_t desc,
                                             void *workspace,
                                             size_t workspace_size,
                                             void *output,
                                             const void *input,
                                             const void *index,
                                             const void *source,
                                             void *stream);

__C __export infiniStatus_t infiniopDestroyIndexCopyDescriptor(infiniopIndexCopyDescriptor_t desc);

#endif