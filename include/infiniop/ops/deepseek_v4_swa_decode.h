#ifndef __INFINIOP_DEEPSEEK_V4_SWA_DECODE_API_H__
#define __INFINIOP_DEEPSEEK_V4_SWA_DECODE_API_H__
#include "../operator_descriptor.h"
typedef struct InfiniopDescriptor *infiniopDeepseekV4SwaDecodeDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateDeepseekV4SwaDecodeDescriptor(infiniopHandle_t, infiniopDeepseekV4SwaDecodeDescriptor_t *, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, float);
__INFINI_C __export infiniStatus_t infiniopGetDeepseekV4SwaDecodeWorkspaceSize(infiniopDeepseekV4SwaDecodeDescriptor_t, size_t *);
__INFINI_C __export infiniStatus_t infiniopDeepseekV4SwaDecode(infiniopDeepseekV4SwaDecodeDescriptor_t, void *, size_t, void *, const void *, const void *, const void *, void *);
__INFINI_C __export infiniStatus_t infiniopDestroyDeepseekV4SwaDecodeDescriptor(infiniopDeepseekV4SwaDecodeDescriptor_t);
#endif
