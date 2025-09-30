#ifndef __INFINIOP_ALL_EQUAL_API_H__
#define __INFINIOP_ALL_EQUAL_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAllEqualDescriptor_t;

__C __export infiniStatus_t infiniopCreateAllEqualDescriptor(
    infiniopHandle_t handle,
    infiniopAllEqualDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc
);

__C __export infiniStatus_t infiniopGetAllEqualWorkspaceSize(infiniopAllEqualDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopAllEqual(
    infiniopAllEqualDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void * c,
    const void * a,
    const void * b,
    void *stream
);

__C __export infiniStatus_t infiniopDestroyAllEqualDescriptor(infiniopAllEqualDescriptor_t desc);

#endif
