#ifndef __INFINIOP_TRIL_API_H__
#define __INFINIOP_TRIL_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTrilDescriptor_t;

__C __export infiniStatus_t infiniopCreateTrilDescriptor(
    infiniopHandle_t handle,
    infiniopTrilDescriptor_t *desc_ptr,
	infiniopTensorDescriptor_t output_desc,
	infiniopTensorDescriptor_t input_desc,
	int diagonal
);

__C __export infiniStatus_t infiniopGetTrilWorkspaceSize(infiniopTrilDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopTril(infiniopTrilDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
	void * output,
	const void * input,
    void *stream
);

__C __export infiniStatus_t infiniopDestroyTrilDescriptor(infiniopTrilDescriptor_t desc);

#endif
