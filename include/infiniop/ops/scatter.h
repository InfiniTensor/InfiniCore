#ifndef __INFINIOP_SCATTER_API_H__
#define __INFINIOP_SCATTER_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopScatterDescriptor_t;

__C __export infiniStatus_t infiniopCreateScatterDescriptor(
    infiniopHandle_t handle,
    infiniopScatterDescriptor_t *desc_ptr,
	infiniopTensorDescriptor_t output_desc,
	infiniopTensorDescriptor_t input_desc,
	infiniopTensorDescriptor_t index_desc,
	size_t dim
);

__C __export infiniStatus_t infiniopGetScatterWorkspaceSize(infiniopScatterDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopScatter(infiniopScatterDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
	void * output,
	const void * input,
	const void * index,
    void *stream
);

__C __export infiniStatus_t infiniopDestroyScatterDescriptor(infiniopScatterDescriptor_t desc);

#endif
