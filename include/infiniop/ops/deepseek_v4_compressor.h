#ifndef __INFINIOP_DEEPSEEK_V4_COMPRESSOR_API_H__
#define __INFINIOP_DEEPSEEK_V4_COMPRESSOR_API_H__

#include "../operator_descriptor.h"

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct InfiniopDescriptor *infiniopDeepseekV4CompressorDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDeepseekV4CompressorDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4CompressorDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t kv_desc,
    infiniopTensorDescriptor_t score_desc,
    infiniopTensorDescriptor_t ape_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    size_t compress_ratio,
    float epsilon);

__INFINI_C __export infiniStatus_t infiniopGetDeepseekV4CompressorWorkspaceSize(
    infiniopDeepseekV4CompressorDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopDeepseekV4Compressor(
    infiniopDeepseekV4CompressorDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *kv,
    const void *score,
    const void *ape,
    const void *norm_weight,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDeepseekV4CompressorDescriptor(
    infiniopDeepseekV4CompressorDescriptor_t desc);

#if defined(__cplusplus)
}
#endif

#endif
