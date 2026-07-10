#ifndef __INFINIOP_DEEPSEEK_V4_SWA_PREFILL_API_H__
#define __INFINIOP_DEEPSEEK_V4_SWA_PREFILL_API_H__
#include "../operator_descriptor.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
typedef struct InfiniopDescriptor *infiniopDeepseekV4SwaPrefillDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateDeepseekV4SwaPrefillDescriptor(infiniopHandle_t, infiniopDeepseekV4SwaPrefillDescriptor_t *, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, float, size_t, size_t, double, bool, double, double, double, int64_t, double);
__INFINI_C __export infiniStatus_t infiniopGetDeepseekV4SwaPrefillWorkspaceSize(infiniopDeepseekV4SwaPrefillDescriptor_t, size_t *);
__INFINI_C __export infiniStatus_t infiniopDeepseekV4SwaPrefill(infiniopDeepseekV4SwaPrefillDescriptor_t, void *, size_t, void *, const void *, const void *, const void *, const void *, const void *, void *);
__INFINI_C __export infiniStatus_t infiniopDestroyDeepseekV4SwaPrefillDescriptor(infiniopDeepseekV4SwaPrefillDescriptor_t);
#endif
