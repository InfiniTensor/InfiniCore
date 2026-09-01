#ifndef __PAGED_CACHING_INFO_H__
#define __PAGED_CACHING_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <optional>
#include <vector>

namespace op::paged_caching {

class PagedCachingInfo {
    PagedCachingInfo() = default;

public:
    // --- Data Type ---
    infiniDtype_t dtype;
    infiniDtype_t cache_dtype;

    // --- Shape Dimensions ---
    size_t num_tokens;
    size_t num_kv_heads;
    size_t head_size;
    size_t v_head_size;
    size_t block_size;

    // --- Strides for Memory Layout ---
    ptrdiff_t k_src_stride;
    ptrdiff_t v_src_stride;
    ptrdiff_t k_src_head_stride;
    ptrdiff_t v_src_head_stride;
    ptrdiff_t k_cache_block_stride;
    ptrdiff_t v_cache_block_stride;
    ptrdiff_t k_cache_head_stride;
    ptrdiff_t v_cache_head_stride;
    ptrdiff_t k_cache_slot_stride;
    ptrdiff_t v_cache_slot_stride;

    // --- Strides for the per-token dequant scales ([num_blocks, num_kv_heads, block_size]) ---
    // Only meaningful when cache_dtype == INFINI_DTYPE_F8.
    ptrdiff_t k_scale_block_stride;
    ptrdiff_t k_scale_head_stride;
    ptrdiff_t k_scale_slot_stride;
    ptrdiff_t v_scale_block_stride;
    ptrdiff_t v_scale_head_stride;
    ptrdiff_t v_scale_slot_stride;

    static utils::Result<PagedCachingInfo> create(
        infiniopTensorDescriptor_t k_cache_desc,
        infiniopTensorDescriptor_t v_cache_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t v_desc,
        infiniopTensorDescriptor_t slot_mapping_desc,
        infiniopTensorDescriptor_t k_scale_desc,
        infiniopTensorDescriptor_t v_scale_desc) {

        auto dtype = k_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        if (v_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        // The caches either keep the source dtype (plain copy) or store FP8(E4M3)
        // codes plus per-token F32 scales (dynamic quantization on write).
        auto cache_dtype = k_cache_desc->dtype();
        const bool cache_fp8 = (cache_dtype == INFINI_DTYPE_F8);
        if (cache_fp8) {
            CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);
            if (v_cache_desc->dtype() != INFINI_DTYPE_F8) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
            if (k_scale_desc == nullptr || v_scale_desc == nullptr) {
                printf("F8 paged_caching requires k_scale and v_scale.\n");
                return INFINI_STATUS_BAD_PARAM;
            }
        } else {
            if (cache_dtype != dtype || v_cache_desc->dtype() != dtype) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
            if (k_scale_desc != nullptr || v_scale_desc != nullptr) {
                printf("k_scale/v_scale are only valid for F8 caches.\n");
                return INFINI_STATUS_BAD_PARAM;
            }
        }
        if (slot_mapping_desc->dtype() != INFINI_DTYPE_I64) {
            printf("slot_mapping must be int64_t.\n");
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (k_desc->ndim() != 3 || v_desc->ndim() != 3 || k_cache_desc->ndim() < 4 || v_cache_desc->ndim() < 4 || slot_mapping_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // PagedCachingInfo info;
        // --- Extract shape dimensions ---
        auto k_shape = k_desc->shape();
        auto v_shape = v_desc->shape();
        auto k_cache_shape = k_cache_desc->shape();
        auto v_cache_shape = v_cache_desc->shape();

        size_t num_tokens = slot_mapping_desc->shape()[0];
        size_t num_kv_heads = k_shape[1];
        size_t head_size = k_shape[2];
        size_t v_head_size = v_shape[2];
        size_t block_size = k_cache_shape[2]; // Assuming [num_blocks, num_heads, block_size, head_size] layout

        if (k_shape[0] != num_tokens || v_shape[0] != num_tokens || v_shape[1] != num_kv_heads) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (k_cache_shape.size() < 4 || v_cache_shape.size() < 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (v_cache_shape[0] != k_cache_shape[0] || v_cache_shape[1] != num_kv_heads || v_cache_shape[2] != block_size) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        // The V cache may be wider than the incoming V. This is used by models
        // that pad V to the Q/K head width for the attention backend.
        if (k_cache_shape[1] != num_kv_heads || k_cache_shape[3] != head_size || v_cache_shape[3] < v_head_size) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // --- Validate per-token scale tensors for the FP8 path ---
        if (cache_fp8) {
            const size_t num_blocks = k_cache_shape[0];
            for (auto scale_desc : {k_scale_desc, v_scale_desc}) {
                if (scale_desc->dtype() != INFINI_DTYPE_F32) {
                    return INFINI_STATUS_BAD_TENSOR_DTYPE;
                }
                if (scale_desc->ndim() != 3) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
                const auto scale_shape = scale_desc->shape();
                if (scale_shape[0] != num_blocks || scale_shape[1] != num_kv_heads || scale_shape[2] != block_size) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            }
        }

        // --- Extract strides for memory access ---
        ptrdiff_t k_src_stride = k_desc->stride(0);
        ptrdiff_t v_src_stride = v_desc->stride(0);
        ptrdiff_t k_src_head_stride = k_desc->stride(1);
        ptrdiff_t v_src_head_stride = v_desc->stride(1);
        ptrdiff_t k_cache_block_stride = k_cache_desc->stride(0);
        ptrdiff_t v_cache_block_stride = v_cache_desc->stride(0);
        ptrdiff_t k_cache_head_stride = k_cache_desc->stride(1);
        ptrdiff_t v_cache_head_stride = v_cache_desc->stride(1);
        ptrdiff_t k_cache_slot_stride = k_cache_desc->stride(2);
        ptrdiff_t v_cache_slot_stride = v_cache_desc->stride(2);

        ptrdiff_t k_scale_block_stride = cache_fp8 ? k_scale_desc->stride(0) : 0;
        ptrdiff_t k_scale_head_stride = cache_fp8 ? k_scale_desc->stride(1) : 0;
        ptrdiff_t k_scale_slot_stride = cache_fp8 ? k_scale_desc->stride(2) : 0;
        ptrdiff_t v_scale_block_stride = cache_fp8 ? v_scale_desc->stride(0) : 0;
        ptrdiff_t v_scale_head_stride = cache_fp8 ? v_scale_desc->stride(1) : 0;
        ptrdiff_t v_scale_slot_stride = cache_fp8 ? v_scale_desc->stride(2) : 0;

        return utils::Result<PagedCachingInfo>(PagedCachingInfo{
            dtype,
            cache_dtype,
            num_tokens,
            num_kv_heads,
            head_size,
            v_head_size,
            block_size,
            k_src_stride,
            v_src_stride,
            k_src_head_stride,
            v_src_head_stride,
            k_cache_block_stride,
            v_cache_block_stride,
            k_cache_head_stride,
            v_cache_head_stride,
            k_cache_slot_stride,
            v_cache_slot_stride,
            k_scale_block_stride,
            k_scale_head_stride,
            k_scale_slot_stride,
            v_scale_block_stride,
            v_scale_head_stride,
            v_scale_slot_stride});
    }
};

} // namespace op::paged_caching

#endif // __PAGED_CACHING_INFO_H__
