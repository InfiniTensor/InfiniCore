#ifndef __INFINIOP_BATCH_NORM_BACKWARD_API_H__
#define __INFINIOP_BATCH_NORM_BACKWARD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopBatchNormBackwardDescriptor_t;

__C __export infiniStatus_t infiniopCreateBatchNormBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopBatchNormBackwardDescriptor_t *desc_ptr,
	infiniopTensorDescriptor_t grad_input_desc,
	infiniopTensorDescriptor_t grad_weight_desc,
	infiniopTensorDescriptor_t grad_bias_desc,
	infiniopTensorDescriptor_t input_desc,
	infiniopTensorDescriptor_t grad_output_desc,
	infiniopTensorDescriptor_t weight_desc,
	infiniopTensorDescriptor_t running_mean_desc,
	infiniopTensorDescriptor_t running_var_desc
);

__C __export infiniStatus_t infiniopGetBatchNormBackwardWorkspaceSize(infiniopBatchNormBackwardDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopBatchNormBackward(infiniopBatchNormBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
	void * grad_input,
	void * grad_weight,
	void * grad_bias,
	const void * input,
	const void * grad_output,
	const void * weight,
	const void * running_mean,
	const void * running_var,
    void *stream
);

__C __export infiniStatus_t infiniopDestroyBatchNormBackwardDescriptor(infiniopBatchNormBackwardDescriptor_t desc);

#endif
