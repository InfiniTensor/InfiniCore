#ifndef __LINEAR_GGUF_INFO_H__
#define __LINEAR_GGUF_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "ggml_blocks.h"

#include <cstdint>

namespace op::linear_gguf {

class LinearGgufInfo {
    LinearGgufInfo() = default;

public:
    size_t m_count;       // batch size
    size_t n_count;       // output features
    size_t k_count;       // input features, in elements (not bytes)
    int32_t ggml_type;    // enum ggml_type id of the weight blocks
    int32_t block_elems;  // elements per block, from ggml_blocks
    int32_t block_bytes;  // bytes per block, from ggml_blocks
    int64_t row_bytes;    // packed bytes of one weight row
    bool out_is_f32;      // experimental decode path: keep GEMV accumulator as F32

    static utils::Result<LinearGgufInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t w_desc,
        int64_t ggml_type) {
        CHECK_OR_RETURN(out_desc != nullptr && a_desc != nullptr && w_desc != nullptr,
                        INFINI_STATUS_NULL_POINTER);

        CHECK_DTYPE(a_desc->dtype(), INFINI_DTYPE_BF16);
        CHECK_DTYPE(out_desc->dtype(), INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        // The weight is block bytes, so its "features" are bytes, not elements.
        CHECK_DTYPE(w_desc->dtype(), INFINI_DTYPE_U8);

        CHECK_OR_RETURN(a_desc->ndim() == 2 && out_desc->ndim() == 2
                            && w_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(a_desc->isContiguous() && out_desc->isContiguous()
                            && w_desc->isContiguous(),
                        INFINI_STATUS_BAD_TENSOR_STRIDES);

        CHECK_OR_RETURN(ggml_type >= 0 && ggml_type <= INT32_MAX,
                        INFINI_STATUS_BAD_PARAM);
        const int32_t type = static_cast<int32_t>(ggml_type);
        const int32_t elems = ggml_blocks::block_elems(type);
        const int32_t bytes = ggml_blocks::block_bytes(type);
        // Every type without a decoder reports 0 / -1 here, so it is rejected
        // the same way a nonexistent id is.
        CHECK_OR_RETURN(elems > 0 && bytes > 0, INFINI_STATUS_BAD_PARAM);

        const size_t m_count = a_desc->dim(0);
        const size_t k_count = a_desc->dim(1);
        const size_t n_count = w_desc->dim(0);
        const int64_t packed = static_cast<int64_t>(w_desc->dim(1));

        CHECK_OR_RETURN(m_count > 0 && n_count > 0 && k_count > 0,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(out_desc->dim(0) == m_count && out_desc->dim(1) == n_count,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(k_count % static_cast<size_t>(elems) == 0,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        // Exact row size: a row may not carry padding past its last block.
        CHECK_OR_RETURN(packed == static_cast<int64_t>(k_count / elems) * bytes,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        LinearGgufInfo info;
        info.m_count = m_count;
        info.n_count = n_count;
        info.k_count = k_count;
        info.ggml_type = type;
        info.block_elems = elems;
        info.block_bytes = bytes;
        info.row_bytes = packed;
        info.out_is_f32 = out_desc->dtype() == INFINI_DTYPE_F32;
        return utils::Result<LinearGgufInfo>(info);
    }
};

} // namespace op::linear_gguf

#endif
