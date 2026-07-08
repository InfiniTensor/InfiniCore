#ifndef __INFINIOP_DSV4_ACT_QUANT_FP8_API_H__
#define __INFINIOP_DSV4_ACT_QUANT_FP8_API_H__
#include "../operator_descriptor.h"
typedef struct InfiniopDescriptor *infiniopDsv4ActQuantFp8Descriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateDsv4ActQuantFp8Descriptor(infiniopHandle_t, infiniopDsv4ActQuantFp8Descriptor_t *, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, float);
__INFINI_C __export infiniStatus_t infiniopGetDsv4ActQuantFp8WorkspaceSize(infiniopDsv4ActQuantFp8Descriptor_t, size_t *);
__INFINI_C __export infiniStatus_t infiniopDsv4ActQuantFp8(infiniopDsv4ActQuantFp8Descriptor_t, void *, size_t, void *, void *, const void *, void *);
__INFINI_C __export infiniStatus_t infiniopDestroyDsv4ActQuantFp8Descriptor(infiniopDsv4ActQuantFp8Descriptor_t);
#endif
