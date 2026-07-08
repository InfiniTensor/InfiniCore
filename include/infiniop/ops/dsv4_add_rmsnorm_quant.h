#ifndef __INFINIOP_DSV4_ADD_RMSNORM_QUANT_API_H__
#define __INFINIOP_DSV4_ADD_RMSNORM_QUANT_API_H__
#include "../operator_descriptor.h"
typedef struct InfiniopDescriptor *infiniopDsv4AddRMSNormQuantDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateDsv4AddRMSNormQuantDescriptor(infiniopHandle_t, infiniopDsv4AddRMSNormQuantDescriptor_t *, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, float);
__INFINI_C __export infiniStatus_t infiniopGetDsv4AddRMSNormQuantWorkspaceSize(infiniopDsv4AddRMSNormQuantDescriptor_t, size_t *);
__INFINI_C __export infiniStatus_t infiniopDsv4AddRMSNormQuant(infiniopDsv4AddRMSNormQuantDescriptor_t, void *, size_t, void *, void *, void *, const void *, const void *, void *);
__INFINI_C __export infiniStatus_t infiniopDestroyDsv4AddRMSNormQuantDescriptor(infiniopDsv4AddRMSNormQuantDescriptor_t);
#endif
