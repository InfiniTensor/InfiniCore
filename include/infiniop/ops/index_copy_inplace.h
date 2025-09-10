#ifndef __INFINIOP_INDEX_COPY_INPLACE_API_H__
#define __INFINIOP_INDEX_COPY_INPLACE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopIndexCopyInplaceDescriptor_t;

__C __export infiniStatus_t infiniopCreateIndexCopyInplaceDescriptor(
    infiniopHandle_t handle,
    infiniopIndexCopyInplaceDescriptor_t *desc_ptr,
	infiniopTensorDescriptor_t output_desc,
	infiniopTensorDescriptor_t input_desc,
	infiniopTensorDescriptor_t index_desc,
	size_t dim
);

__C __export infiniStatus_t infiniopGetIndexCopyInplaceWorkspaceSize(infiniopIndexCopyInplaceDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopIndexCopyInplace(infiniopIndexCopyInplaceDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
	void * output,
	const void * input,
	const void * index,
    void *stream
);

__C __export infiniStatus_t infiniopDestroyIndexCopyInplaceDescriptor(infiniopIndexCopyInplaceDescriptor_t desc);

#endif
