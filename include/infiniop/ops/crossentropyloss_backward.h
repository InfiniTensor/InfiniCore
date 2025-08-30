#ifndef __INFINIOP_CROSSENTROPYLOSS_BACKWARD_API_H__
#define __INFINIOP_CROSSENTROPYLOSS_BACKWARD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopCrossEntropyLossBackWardDescriptor_t;

__C __export infiniStatus_t infiniopCreateCrossEntropyLossBackWardDescriptor(infiniopHandle_t handle,
                                                                             infiniopCrossEntropyLossBackWardDescriptor_t *desc_ptr,
                                                                             infiniopTensorDescriptor_t grad_logits,
                                                                             infiniopTensorDescriptor_t probs,
                                                                             infiniopTensorDescriptor_t target);

__C __export infiniStatus_t infiniopGetCrossEntropyLossBackWardWorkspaceSize(infiniopCrossEntropyLossBackWardDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopCrossEntropyLossBackWard(infiniopCrossEntropyLossBackWardDescriptor_t desc,
                                                             void *workspace,
                                                             size_t workspace_size,
                                                             void *grad_logits,
                                                             const void *probs,
                                                             const void *target,
                                                             void *stream);

__C __export infiniStatus_t infiniopDestroyCrossEntropyLossBackWardDescriptor(infiniopCrossEntropyLossBackWardDescriptor_t desc);

#endif
