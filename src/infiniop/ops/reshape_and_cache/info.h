#ifndef __RESHAPE_AND_CACHE_INFO_H__
#define __RESHAPE_AND_CACHE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <optional>
#include <vector>

namespace op::reshape_and_cache {

class ReshapeAndCacheInfo {
    ReshapeAndCacheInfo() = default;

public:
    // --- Data Type ---
    infiniDtype_t dtype;

    size_t num_tokens;
    size_t num_kv_heads;
    size_t head_size;
    size_t block_size;
    size_t x;

    ptrdiff_t key_stride;
    ptrdiff_t value_stride;

    static utils::Result<ReshapeAndCacheInfo> create(
        infiniopTensorDescriptor_t key_desc,          // [num_tokens, num_heads, head_size]
        infiniopTensorDescriptor_t value_desc,        // [num_tokens, num_heads, head_size]
        infiniopTensorDescriptor_t key_cache_desc,    // [num_blocks, num_heads, head_size/x, block_size, x]
        infiniopTensorDescriptor_t value_cache_desc,  // [num_blocks, num_heads, head_size, block_size]
        infiniopTensorDescriptor_t slot_mapping_desc, // [num_tokens]
        const char *kv_cache_dtype) {

        auto dtype = key_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        if (value_desc->dtype() != dtype || key_cache_desc->dtype() != dtype || value_cache_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (slot_mapping_desc->dtype() != INFINI_DTYPE_I64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (key_desc->ndim() != 3 || value_desc->ndim() != 3 || key_cache_desc->ndim() < 4 || value_cache_desc->ndim() < 4 || slot_mapping_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t num_tokens = slot_mapping_desc->shape()[0];
        size_t num_kv_heads = key_desc->shape()[1];
        size_t head_size = key_desc->shape()[2];
        size_t block_size = key_cache_desc->shape()[3];
        size_t x = key_cache_desc->shape()[4];

        ptrdiff_t key_stride = key_desc->stride(0);
        ptrdiff_t value_stride = value_desc->stride(0);

        return utils::Result<ReshapeAndCacheInfo>(ReshapeAndCacheInfo{
            dtype,
            num_tokens,
            num_kv_heads,
            head_size,
            block_size,
            x,
            key_stride,
            value_stride});
    }
};

} // namespace op::reshape_and_cache

#endif // __RESHAPE_AND_CACHE_INFO_H__
