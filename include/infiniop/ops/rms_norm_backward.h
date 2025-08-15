#ifndef __INFINIOP_RMS_NORM_BACKWARD_API_H__
#define __INFINIOP_RMS_NORM_BACKWARD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRMSNormBackwardDescriptor_t;

__C __export infiniStatus_t infiniopCreateRMSNormBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopRMSNormBackwardDescriptor_t *desc_ptr,
	infiniopTensorDescriptor_t grad_x_desc,
	infiniopTensorDescriptor_t grad_w_desc,
	infiniopTensorDescriptor_t grad_y_desc,
	infiniopTensorDescriptor_t x_desc,
	infiniopTensorDescriptor_t w_desc
);

__C __export infiniStatus_t infiniopGetRMSNormBackwardWorkspaceSize(infiniopRMSNormBackwardDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopRMSNormBackward(infiniopRMSNormBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
	void * grad_x,
	void * grad_w,
	const void * grad_y,
	const void * x,
	const void * w,
    void *stream
);

__C __export infiniStatus_t infiniopDestroyRMSNormBackwardDescriptor(infiniopRMSNormBackwardDescriptor_t desc);

#endif
