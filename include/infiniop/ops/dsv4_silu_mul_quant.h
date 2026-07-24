#ifndef __INFINIOP_DSV4_SILU_MUL_QUANT_API_H__
#define __INFINIOP_DSV4_SILU_MUL_QUANT_API_H__
#include "../operator_descriptor.h"
typedef struct InfiniopDescriptor *infiniopDsv4SiluMulQuantDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateDsv4SiluMulQuantDescriptor(infiniopHandle_t, infiniopDsv4SiluMulQuantDescriptor_t *, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t);
__INFINI_C __export infiniStatus_t infiniopGetDsv4SiluMulQuantWorkspaceSize(infiniopDsv4SiluMulQuantDescriptor_t, size_t *);
__INFINI_C __export infiniStatus_t infiniopDsv4SiluMulQuant(infiniopDsv4SiluMulQuantDescriptor_t, void *, size_t, void *, void *, const void *, const void *, void *);
__INFINI_C __export infiniStatus_t infiniopDestroyDsv4SiluMulQuantDescriptor(infiniopDsv4SiluMulQuantDescriptor_t);
#endif
