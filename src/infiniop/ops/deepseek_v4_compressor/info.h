#ifndef __DEEPSEEK_V4_COMPRESSOR_INFO_H__
#define __DEEPSEEK_V4_COMPRESSOR_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

#include <cstddef>

namespace op::deepseek_v4_compressor {

struct DeepseekV4CompressorInfo {
    size_t batch_size;
    size_t seq_len;
    size_t num_blocks;
    size_t head_dim;
    size_t compressed_dim;
    size_t compress_ratio;
    size_t coff;
    float epsilon;
    infiniDtype_t dtype;

    static utils::Result<DeepseekV4CompressorInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t kv_desc,
        infiniopTensorDescriptor_t score_desc,
        infiniopTensorDescriptor_t ape_desc,
        infiniopTensorDescriptor_t norm_weight_desc,
        size_t compress_ratio,
        float epsilon) {
        auto dtype = kv_desc->dtype();
        if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_BF16 && dtype != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (score_desc->dtype() != dtype || ape_desc->dtype() != dtype || out_desc->dtype() != dtype
            || norm_weight_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (kv_desc->ndim() != 3 || score_desc->ndim() != 3 || ape_desc->ndim() != 2
            || norm_weight_desc->ndim() != 1 || out_desc->ndim() != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (compress_ratio == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t batch_size = kv_desc->shape()[0];
        const size_t seq_len = kv_desc->shape()[1];
        const size_t compressed_dim = kv_desc->shape()[2];
        const size_t head_dim = norm_weight_desc->shape()[0];
        const size_t num_blocks = seq_len / compress_ratio;
        if (batch_size == 0 || seq_len == 0 || compressed_dim == 0 || head_dim == 0 || num_blocks == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (compressed_dim % head_dim != 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t coff = compressed_dim / head_dim;
        if (coff != 1 && coff != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (score_desc->shape()[0] != batch_size || score_desc->shape()[1] != seq_len
            || score_desc->shape()[2] != compressed_dim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (ape_desc->shape()[0] != compress_ratio || ape_desc->shape()[1] != compressed_dim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (out_desc->shape()[0] != batch_size || out_desc->shape()[1] != num_blocks
            || out_desc->shape()[2] != head_dim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (!kv_desc->isContiguous() || !score_desc->isContiguous() || !ape_desc->isContiguous()
            || !norm_weight_desc->isContiguous() || !out_desc->isContiguous()) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        if (head_dim > 4096) {
            return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
        }
        return utils::Result<DeepseekV4CompressorInfo>(DeepseekV4CompressorInfo{
            batch_size,
            seq_len,
            num_blocks,
            head_dim,
            compressed_dim,
            compress_ratio,
            coff,
            epsilon,
            dtype,
        });
    }
};

} // namespace op::deepseek_v4_compressor

#endif
