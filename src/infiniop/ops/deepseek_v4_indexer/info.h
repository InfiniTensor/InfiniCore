#ifndef __DEEPSEEK_V4_INDEXER_INFO_H__
#define __DEEPSEEK_V4_INDEXER_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

#include <cmath>
#include <cstddef>

namespace op::deepseek_v4_indexer {

struct DeepseekV4IndexerInfo {
    size_t batch_size;
    size_t query_len;
    size_t index_n_heads;
    size_t head_dim;
    size_t num_blocks;
    size_t topk;
    size_t query_start;
    size_t compress_ratio;
    float score_scale;
    float weight_scale;
    infiniDtype_t dtype;

    static utils::Result<DeepseekV4IndexerInfo> create(
        infiniopTensorDescriptor_t indices_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t weights_desc,
        infiniopTensorDescriptor_t compressed_desc,
        infiniopTensorDescriptor_t positions_desc,
        size_t query_start,
        size_t compress_ratio) {
        if (indices_desc->dtype() != INFINI_DTYPE_I64 || positions_desc->dtype() != INFINI_DTYPE_I64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        auto dtype = q_desc->dtype();
        if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_BF16 && dtype != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (weights_desc->dtype() != dtype || compressed_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (q_desc->ndim() != 4 || weights_desc->ndim() != 3 || compressed_desc->ndim() != 3
            || indices_desc->ndim() != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t batch_size = q_desc->shape()[0];
        const size_t query_len = q_desc->shape()[1];
        const size_t index_n_heads = q_desc->shape()[2];
        const size_t head_dim = q_desc->shape()[3];
        const size_t num_blocks = compressed_desc->shape()[1];
        const size_t topk = indices_desc->shape()[2];
        if (batch_size == 0 || query_len == 0 || index_n_heads == 0 || head_dim == 0
            || num_blocks == 0 || topk == 0 || compress_ratio == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (weights_desc->shape()[0] != batch_size || weights_desc->shape()[1] != query_len
            || weights_desc->shape()[2] != index_n_heads) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (compressed_desc->shape()[0] != batch_size || compressed_desc->shape()[2] != head_dim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (indices_desc->shape()[0] != batch_size || indices_desc->shape()[1] != query_len
            || topk > num_blocks) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (positions_desc->numel() < query_start + query_len) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (!q_desc->isContiguous() || !weights_desc->isContiguous() || !compressed_desc->isContiguous()
            || !indices_desc->isContiguous() || !positions_desc->isContiguous()) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        if (topk > 1024) {
            return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
        }
        return utils::Result<DeepseekV4IndexerInfo>(DeepseekV4IndexerInfo{
            batch_size,
            query_len,
            index_n_heads,
            head_dim,
            num_blocks,
            topk,
            query_start,
            compress_ratio,
            1.0f / std::sqrt(static_cast<float>(head_dim)),
            1.0f / std::sqrt(static_cast<float>(index_n_heads)),
            dtype,
        });
    }
};

} // namespace op::deepseek_v4_indexer

#endif
