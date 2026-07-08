#ifndef __INFINIOP_DSV4_SILU_AND_MUL_API_H__
#define __INFINIOP_DSV4_SILU_AND_MUL_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4SiluAndMulDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4SiluAndMulDescriptor(infiniopHandle_t handle, infiniopDsv4SiluAndMulDescriptor_t *desc_ptr, infiniopTensorDescriptor_t y_desc, infiniopTensorDescriptor_t gate_desc, infiniopTensorDescriptor_t up_desc);
__INFINI_C __export infiniStatus_t infiniopGetDsv4SiluAndMulWorkspaceSize(infiniopDsv4SiluAndMulDescriptor_t desc, size_t *size);
__INFINI_C __export infiniStatus_t infiniopDsv4SiluAndMul(infiniopDsv4SiluAndMulDescriptor_t desc, void *workspace, size_t workspace_size, void *y, const void *gate, const void *up, void *stream);
__INFINI_C __export infiniStatus_t infiniopDestroyDsv4SiluAndMulDescriptor(infiniopDsv4SiluAndMulDescriptor_t desc);

#endif // __INFINIOP_DSV4_SILU_AND_MUL_API_H__
