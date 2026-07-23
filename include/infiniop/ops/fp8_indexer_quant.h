#ifndef __INFINIOP_FP8_INDEXER_QUANT_API_H__
#define __INFINIOP_FP8_INDEXER_QUANT_API_H__

#include "../operator_descriptor.h"

#include <stdint.h>

typedef struct InfiniopDescriptor *infiniopFp8IndexerQuantDescriptor_t;
typedef struct InfiniopDescriptor *infiniopFusedFp8IndexerDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFp8IndexerQuantDescriptor(
    infiniopHandle_t handle,
    infiniopFp8IndexerQuantDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t q_fp8_desc,
    infiniopTensorDescriptor_t weights_fp32_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t weights_desc);

__INFINI_C __export infiniStatus_t infiniopFp8IndexerQuant(
    infiniopFp8IndexerQuantDescriptor_t desc,
    void *q_fp8,
    void *weights_fp32,
    const void *q,
    const void *weights,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFp8IndexerQuantDescriptor(
    infiniopFp8IndexerQuantDescriptor_t desc);

__INFINI_C __export infiniStatus_t infiniopCreateFusedFp8IndexerDescriptor(
    infiniopHandle_t handle,
    infiniopFusedFp8IndexerDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t q_fp8_desc,
    infiniopTensorDescriptor_t weights_fp32_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t q_raw_desc,
    infiniopTensorDescriptor_t k_weights_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    infiniopTensorDescriptor_t norm_bias_desc,
    infiniopTensorDescriptor_t positions_desc,
    infiniopTensorDescriptor_t cos_sin_cache_desc,
    infiniopTensorDescriptor_t slot_mapping_desc,
    uint64_t rope_dim,
    double eps,
    double weights_scale);

__INFINI_C __export infiniStatus_t infiniopFusedFp8Indexer(
    infiniopFusedFp8IndexerDescriptor_t desc,
    void *q_fp8,
    void *weights_fp32,
    void *k_cache,
    const void *q_raw,
    const void *k_weights,
    const void *norm_weight,
    const void *norm_bias,
    const void *positions,
    const void *cos_sin_cache,
    const void *slot_mapping,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFusedFp8IndexerDescriptor(
    infiniopFusedFp8IndexerDescriptor_t desc);

#endif
