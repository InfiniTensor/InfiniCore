#ifndef __INFINIOP_DSV4_RMSNORM_SELF_API_H__
#define __INFINIOP_DSV4_RMSNORM_SELF_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4RMSNormSelfDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4RMSNormSelfDescriptor(
    infiniopHandle_t handle,
    infiniopDsv4RMSNormSelfDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    float epsilon);
__INFINI_C __export infiniStatus_t infiniopGetDsv4RMSNormSelfWorkspaceSize(infiniopDsv4RMSNormSelfDescriptor_t desc, size_t *size);
__INFINI_C __export infiniStatus_t infiniopDsv4RMSNormSelf(infiniopDsv4RMSNormSelfDescriptor_t desc, void *workspace, size_t workspace_size, void *y, const void *x, void *stream);
__INFINI_C __export infiniStatus_t infiniopDestroyDsv4RMSNormSelfDescriptor(infiniopDsv4RMSNormSelfDescriptor_t desc);

#endif // __INFINIOP_DSV4_RMSNORM_SELF_API_H__
