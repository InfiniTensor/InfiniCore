#ifndef __INFINIOP_BATCH_NORM_API_H__
#define __INFINIOP_BATCH_NORM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopBatchNormDescriptor_t;

__C __export infiniStatus_t infiniopCreateBatchNormDescriptor(
    infiniopHandle_t handle,
    infiniopBatchNormDescriptor_t *desc_ptr,
	infiniopTensorDescriptor_t output_desc,
	infiniopTensorDescriptor_t running_mean_desc,
	infiniopTensorDescriptor_t running_var_desc,
	infiniopTensorDescriptor_t input_desc,
	infiniopTensorDescriptor_t weight_desc,
	infiniopTensorDescriptor_t bias_desc,
	float momentum,
	float eps
);

__C __export infiniStatus_t infiniopGetBatchNormWorkspaceSize(infiniopBatchNormDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopBatchNorm(infiniopBatchNormDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
	void * output,
	void * running_mean,
	void * running_var,
	const void * input,
	const void * weight,
	const void * bias,
    void *stream
);

__C __export infiniStatus_t infiniopDestroyBatchNormDescriptor(infiniopBatchNormDescriptor_t desc);

#endif
