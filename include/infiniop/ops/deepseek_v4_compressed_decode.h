#ifndef __INFINIOP_DEEPSEEK_V4_COMPRESSED_DECODE_API_H__
#define __INFINIOP_DEEPSEEK_V4_COMPRESSED_DECODE_API_H__
#include "../operator_descriptor.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
typedef struct InfiniopDescriptor *infiniopDeepseekV4CompressedDecodeDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateDeepseekV4CompressedDecodeDescriptor(infiniopHandle_t, infiniopDeepseekV4CompressedDecodeDescriptor_t *, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t, float, size_t, size_t, double, bool, double, double, double, int64_t, double);
__INFINI_C __export infiniStatus_t infiniopGetDeepseekV4CompressedDecodeWorkspaceSize(infiniopDeepseekV4CompressedDecodeDescriptor_t, size_t *);
__INFINI_C __export infiniStatus_t infiniopDeepseekV4CompressedDecode(infiniopDeepseekV4CompressedDecodeDescriptor_t, void *, size_t, void *, const void *, const void *, const void *, const void *, const void *, const void *, void *);
__INFINI_C __export infiniStatus_t infiniopDestroyDeepseekV4CompressedDecodeDescriptor(infiniopDeepseekV4CompressedDecodeDescriptor_t);
#endif
