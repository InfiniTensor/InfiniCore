#ifndef __FP8_INDEXER_QUANT_NVIDIA_H__
#define __FP8_INDEXER_QUANT_NVIDIA_H__

#include "../fp8_indexer_quant.h"

DESCRIPTOR(nvidia)

namespace op::fp8_indexer_quant::nvidia {
class FusedDescriptor final : public InfiniopDescriptor {
    size_t _num_tokens;
    size_t _num_heads;
    size_t _head_dim;
    size_t _rope_dim;
    size_t _num_cache_blocks;
    size_t _block_size;
    size_t _cache_stride;
    size_t _max_positions;
    infiniDtype_t _input_dtype;
    float _eps;
    float _weights_scale;

    FusedDescriptor(
        size_t num_tokens,
        size_t num_heads,
        size_t head_dim,
        size_t rope_dim,
        size_t num_cache_blocks,
        size_t block_size,
        size_t cache_stride,
        size_t max_positions,
        infiniDtype_t input_dtype,
        double eps,
        double weights_scale,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _num_tokens(num_tokens),
          _num_heads(num_heads),
          _head_dim(head_dim),
          _rope_dim(rope_dim),
          _num_cache_blocks(num_cache_blocks),
          _block_size(block_size),
          _cache_stride(cache_stride),
          _max_positions(max_positions),
          _input_dtype(input_dtype),
          _eps(static_cast<float>(eps)),
          _weights_scale(static_cast<float>(weights_scale)) {}

public:
    static infiniStatus_t create(
        infiniopHandle_t handle,
        FusedDescriptor **desc_ptr,
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

    infiniStatus_t calculate(
        void *q_fp8, void *weights_fp32, void *k_cache,
        const void *q_raw, const void *k_weights,
        const void *norm_weight, const void *norm_bias,
        const void *positions, const void *cos_sin_cache,
        const void *slot_mapping, void *stream) const;
};
} // namespace op::fp8_indexer_quant::nvidia

#endif
