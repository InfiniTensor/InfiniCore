#ifndef __PAGED_CACHING_PREFILL_INFO_H__
#define __PAGED_CACHING_PREFILL_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::paged_caching_prefill {

class PagedCachingPrefillInfo {
public:
    infiniDtype_t dtype;

    // --- Shape Dimensions ---
    size_t num_tokens;   // 当前 Batch 中新增的 Token 总数
    size_t num_kv_heads;
    size_t head_size;
    size_t block_size;

    // --- Strides ---
    ptrdiff_t k_src_stride;
    ptrdiff_t v_src_stride;
    ptrdiff_t k_cache_block_stride;
    ptrdiff_t v_cache_block_stride;

    static utils::Result<PagedCachingPrefillInfo> create(
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t v_desc,
        infiniopTensorDescriptor_t k_cache_desc,
        infiniopTensorDescriptor_t v_cache_desc,
        infiniopTensorDescriptor_t slot_mapping_desc) {

        auto dtype = k_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        
        if (v_desc->dtype() != dtype || k_cache_desc->dtype() != dtype || v_cache_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (slot_mapping_desc->dtype() != INFINI_DTYPE_I32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        auto k_shape = k_desc->shape();
        auto k_cache_shape = k_cache_desc->shape();

        // Prefill 模式下，num_tokens 由 slot_mapping 的长度决定
        size_t num_tokens = slot_mapping_desc->shape()[0];
        size_t num_kv_heads = k_shape[1];
        size_t head_size = k_shape[2];
        size_t block_size = k_cache_shape[2]; 

        return utils::Result<PagedCachingPrefillInfo>(PagedCachingPrefillInfo{
            dtype,
            num_tokens,
            num_kv_heads,
            head_size,
            block_size,
            k_desc->stride(0),
            v_desc->stride(0),
            k_cache_desc->stride(0),
            v_cache_desc->stride(0)});
    }
};

} // namespace op::paged_caching_prefill

#endif // __PAGED_CACHING_PREFILL_INFO_H__
