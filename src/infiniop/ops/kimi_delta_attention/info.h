#ifndef __KIMI_DELTA_ATTENTION_INFO_H__
#define __KIMI_DELTA_ATTENTION_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

#include <vector>

namespace op::kimi_delta_attention {

class KimiDeltaAttentionInfo {
    KimiDeltaAttentionInfo() = default;

public:
    infiniDtype_t data_dtype;
    infiniDtype_t gate_dtype;
    infiniDtype_t cu_seqlens_dtype;
    infiniDtype_t initial_state_indices_dtype;
    infiniDtype_t final_state_indices_dtype;

    bool has_cu_seqlens;
    bool has_initial_state_indices;
    bool has_final_state_indices;
    bool indexed_state_pool;
    bool is_decode;
    bool use_qk_l2norm;

    size_t B, T, total_tokens, H, D, pool_size;
    float scale;
    float lower_bound;

    std::vector<ptrdiff_t> out_strides;
    std::vector<ptrdiff_t> initial_state_strides;
    std::vector<ptrdiff_t> final_state_strides;
    std::vector<ptrdiff_t> q_strides;
    std::vector<ptrdiff_t> k_strides;
    std::vector<ptrdiff_t> v_strides;
    std::vector<ptrdiff_t> g_strides;
    std::vector<ptrdiff_t> beta_strides;
    std::vector<ptrdiff_t> A_log_strides;
    std::vector<ptrdiff_t> dt_bias_strides;

    static utils::Result<KimiDeltaAttentionInfo>
    create(infiniopTensorDescriptor_t out_desc,
           infiniopTensorDescriptor_t initial_state_desc,
           infiniopTensorDescriptor_t final_state_desc,
           infiniopTensorDescriptor_t q_desc,
           infiniopTensorDescriptor_t k_desc,
           infiniopTensorDescriptor_t v_desc,
           infiniopTensorDescriptor_t g_desc,
           infiniopTensorDescriptor_t beta_desc,
           infiniopTensorDescriptor_t A_log_desc,
           infiniopTensorDescriptor_t dt_bias_desc,
           infiniopTensorDescriptor_t cu_seqlens_desc,
           infiniopTensorDescriptor_t initial_state_indices_desc,
           infiniopTensorDescriptor_t final_state_indices_desc,
           float scale,
           float lower_bound,
           bool use_qk_l2norm) {

        if (out_desc == nullptr || initial_state_desc == nullptr || q_desc == nullptr || k_desc == nullptr || v_desc == nullptr || g_desc == nullptr || beta_desc == nullptr || A_log_desc == nullptr || dt_bias_desc == nullptr) {
            return INFINI_STATUS_NULL_POINTER;
        }

        auto data_dtype = q_desc->dtype();
        CHECK_DTYPE(data_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        if (k_desc->dtype() != data_dtype || v_desc->dtype() != data_dtype || out_desc->dtype() != data_dtype || initial_state_desc->dtype() != data_dtype || (final_state_desc != nullptr && final_state_desc->dtype() != data_dtype)) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        auto gate_dtype = g_desc->dtype();
        CHECK_DTYPE(gate_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        if (beta_desc->dtype() != gate_dtype || A_log_desc->dtype() != INFINI_DTYPE_F32 || dt_bias_desc->dtype() != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        const bool has_cu = cu_seqlens_desc != nullptr;
        const bool has_initial_indices = initial_state_indices_desc != nullptr;
        const bool has_final_indices = final_state_indices_desc != nullptr;
        const bool indexed_pool = has_initial_indices || has_final_indices;
        if (has_final_indices && final_state_desc != nullptr) {
            return INFINI_STATUS_BAD_PARAM;
        }
        if (!has_final_indices && final_state_desc == nullptr) {
            return INFINI_STATUS_NULL_POINTER;
        }

        if (q_desc->ndim() != 4 || k_desc->ndim() != 4 || v_desc->ndim() != 4 || g_desc->ndim() != 4 || out_desc->ndim() != 4 || beta_desc->ndim() != 3 || A_log_desc->ndim() != 1 || dt_bias_desc->ndim() != 2 || initial_state_desc->ndim() != 4 || (!has_final_indices && final_state_desc->ndim() != 4)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto q_shape = q_desc->shape();
        auto k_shape = k_desc->shape();
        auto v_shape = v_desc->shape();
        auto g_shape = g_desc->shape();
        auto out_shape = out_desc->shape();
        auto beta_shape = beta_desc->shape();

        size_t B = q_shape[0], T = q_shape[1], H = q_shape[2], D = q_shape[3], total_tokens = T;
        if (has_cu) {
            if (cu_seqlens_desc->ndim() != 1 || cu_seqlens_desc->shape()[0] < 2) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            B = cu_seqlens_desc->shape()[0] - 1;
            if (q_shape[0] != 1 || k_shape[0] != 1 || v_shape[0] != 1 || g_shape[0] != 1 || out_shape[0] != 1 || beta_shape[0] != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            total_tokens = q_shape[1];
            T = total_tokens;
        }

        if (k_shape != q_shape || v_shape != q_shape || g_shape != q_shape || out_shape != q_shape || beta_shape[0] != q_shape[0] || beta_shape[1] != q_shape[1] || beta_shape[2] != H || H == 0 || D == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (A_log_desc->shape()[0] != H || dt_bias_desc->shape()[0] != H || dt_bias_desc->shape()[1] != D) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (q_desc->strides()[3] != 1 || k_desc->strides()[3] != 1 || v_desc->strides()[3] != 1 || g_desc->strides()[3] != 1 || out_desc->strides()[3] != 1 || dt_bias_desc->strides()[1] != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        auto initial_shape = initial_state_desc->shape();
        size_t pool_size = initial_shape[0];
        if (indexed_pool) {
            if (initial_shape[1] != H || initial_shape[2] != D || initial_shape[3] != D) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        } else if (initial_shape[0] != B || initial_shape[1] != H || initial_shape[2] != D || initial_shape[3] != D) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (!has_final_indices) {
            auto final_shape = final_state_desc->shape();
            if (final_shape[0] != B || final_shape[1] != H || final_shape[2] != D || final_shape[3] != D) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        infiniDtype_t cu_dtype = INFINI_DTYPE_INVALID;
        infiniDtype_t initial_indices_dtype = INFINI_DTYPE_INVALID;
        infiniDtype_t final_indices_dtype = INFINI_DTYPE_INVALID;
        if (has_cu) {
            cu_dtype = cu_seqlens_desc->dtype();
            CHECK_DTYPE(cu_dtype, INFINI_DTYPE_I32, INFINI_DTYPE_I64);
        }
        if (has_initial_indices) {
            if (initial_state_indices_desc->ndim() != 1 || initial_state_indices_desc->shape()[0] != B) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            initial_indices_dtype = initial_state_indices_desc->dtype();
            CHECK_DTYPE(initial_indices_dtype, INFINI_DTYPE_I32, INFINI_DTYPE_I64);
        }
        if (has_final_indices) {
            if (final_state_indices_desc->ndim() != 1 || final_state_indices_desc->shape()[0] != B) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            final_indices_dtype = final_state_indices_desc->dtype();
            CHECK_DTYPE(final_indices_dtype, INFINI_DTYPE_I32, INFINI_DTYPE_I64);
        }

        KimiDeltaAttentionInfo info;
        info.data_dtype = data_dtype;
        info.gate_dtype = gate_dtype;
        info.cu_seqlens_dtype = cu_dtype;
        info.initial_state_indices_dtype = initial_indices_dtype;
        info.final_state_indices_dtype = final_indices_dtype;
        info.has_cu_seqlens = has_cu;
        info.has_initial_state_indices = has_initial_indices;
        info.has_final_state_indices = has_final_indices;
        info.indexed_state_pool = indexed_pool;
        info.is_decode = has_cu ? (total_tokens == B) : (T == 1);
        info.use_qk_l2norm = use_qk_l2norm;
        info.B = B;
        info.T = T;
        info.total_tokens = total_tokens;
        info.H = H;
        info.D = D;
        info.pool_size = pool_size;
        info.scale = scale;
        info.lower_bound = lower_bound;
        info.out_strides = out_desc->strides();
        info.initial_state_strides = initial_state_desc->strides();
        if (final_state_desc != nullptr) {
            info.final_state_strides = final_state_desc->strides();
        }
        info.q_strides = q_desc->strides();
        info.k_strides = k_desc->strides();
        info.v_strides = v_desc->strides();
        info.g_strides = g_desc->strides();
        info.beta_strides = beta_desc->strides();
        info.A_log_strides = A_log_desc->strides();
        info.dt_bias_strides = dt_bias_desc->strides();
        return utils::Result<KimiDeltaAttentionInfo>(info);
    }
};

} // namespace op::kimi_delta_attention

#endif
