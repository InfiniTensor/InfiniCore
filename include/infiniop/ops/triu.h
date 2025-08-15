#ifndef __INFINIOP_TRIU_API_H__
#define __INFINIOP_TRIU_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTriuDescriptor_t;

__C __export infiniStatus_t infiniopCreateTriuDescriptor(
    infiniopHandle_t handle,
    infiniopTriuDescriptor_t *desc_ptr,
	infiniopTensorDescriptor_t output_desc,
	infiniopTensorDescriptor_t input_desc,
	int diagonal
);

__C __export infiniStatus_t infiniopGetTriuWorkspaceSize(infiniopTriuDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopTriu(infiniopTriuDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
	void * output,
	const void * input,
    void *stream
);

__C __export infiniStatus_t infiniopDestroyTriuDescriptor(infiniopTriuDescriptor_t desc);

#endif
