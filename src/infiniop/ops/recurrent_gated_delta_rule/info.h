// infiniop/ops/recurrent_gated_delta_rule/info.h

#ifndef __RECURRENT_GATED_DELTA_RULE_INFO_H__
#define __RECURRENT_GATED_DELTA_RULE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op {
namespace recurrent_gated_delta_rule {

class RecurrentGatedDeltaRuleInfo {
    RecurrentGatedDeltaRuleInfo() = default;

public:
    infiniDtype_t data_dtype;
    infiniDtype_t gate_dtype;
    infiniDtype_t initial_state_indices_dtype;
    infiniDtype_t final_state_indices_dtype;
    bool use_qk_l2norm;
    bool has_initial_state_indices;
    bool has_final_state_indices;
    bool indexed_state_pool;

    size_t B, Hk, Hv, T, Dk, Dv, pool_size, value_heads_per_key_head;

    std::vector<ptrdiff_t> out_strides;
    std::vector<ptrdiff_t> initial_state_strides;
    std::vector<ptrdiff_t> final_state_strides;
    std::vector<ptrdiff_t> q_strides;
    std::vector<ptrdiff_t> k_strides;
    std::vector<ptrdiff_t> v_strides;
    std::vector<ptrdiff_t> g_strides;
    std::vector<ptrdiff_t> beta_strides;

    static utils::Result<RecurrentGatedDeltaRuleInfo>
    create(infiniopTensorDescriptor_t out_desc,
           infiniopTensorDescriptor_t initial_state_desc,
           infiniopTensorDescriptor_t final_state_desc,
           infiniopTensorDescriptor_t q_desc,
           infiniopTensorDescriptor_t k_desc,
           infiniopTensorDescriptor_t v_desc,
           infiniopTensorDescriptor_t g_desc,
           infiniopTensorDescriptor_t beta_desc,
           infiniopTensorDescriptor_t initial_state_indices_desc,
           infiniopTensorDescriptor_t final_state_indices_desc,
           bool use_qk_l2norm) {

        if (out_desc == nullptr || initial_state_desc == nullptr || q_desc == nullptr || k_desc == nullptr || v_desc == nullptr || g_desc == nullptr || beta_desc == nullptr) {
            return INFINI_STATUS_NULL_POINTER;
        }

        auto data_dtype = q_desc->dtype();
        CHECK_DTYPE(data_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        if (k_desc->dtype() != data_dtype || v_desc->dtype() != data_dtype || out_desc->dtype() != data_dtype || initial_state_desc->dtype() != data_dtype || (final_state_desc != nullptr && final_state_desc->dtype() != data_dtype)) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        auto gate_dtype = g_desc->dtype();
        CHECK_DTYPE(gate_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        if (beta_desc->dtype() != gate_dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        bool has_initial_indices = initial_state_indices_desc != nullptr;
        bool has_final_indices = final_state_indices_desc != nullptr;
        bool indexed_pool = has_initial_indices || has_final_indices;

        if (has_final_indices && final_state_desc != nullptr) {
            return INFINI_STATUS_BAD_PARAM;
        }
        if (!has_final_indices && final_state_desc == nullptr) {
            return INFINI_STATUS_NULL_POINTER;
        }

        if (q_desc->ndim() != 4 || k_desc->ndim() != 4 || v_desc->ndim() != 4 || out_desc->ndim() != 4 || g_desc->ndim() != 3 || beta_desc->ndim() != 3 || initial_state_desc->ndim() != 4 || (!has_final_indices && final_state_desc->ndim() != 4)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto q_shape = q_desc->shape();       // [B, T, Hk, Dk]
        auto k_shape = k_desc->shape();       // [B, T, Hk, Dk]
        auto v_shape = v_desc->shape();       // [B, T, Hv, Dv]
        auto out_shape = out_desc->shape();   // [B, T, Hv, Dv]
        auto g_shape = g_desc->shape();       // [B, T, Hv]
        auto beta_shape = beta_desc->shape(); // [B, T, Hv]

        size_t B = q_shape[0], T = q_shape[1], Hk = q_shape[2], Dk = q_shape[3];
        size_t Hv = v_shape[2], Dv = v_shape[3];

        if (T != 1 || k_shape[0] != B || k_shape[1] != T || k_shape[2] != Hk || k_shape[3] != Dk || v_shape[0] != B || v_shape[1] != T || out_shape[0] != B || out_shape[1] != T || out_shape[2] != Hv || out_shape[3] != Dv || g_shape[0] != B || g_shape[1] != T || g_shape[2] != Hv || beta_shape[0] != B || beta_shape[1] != T || beta_shape[2] != Hv || Hk == 0 || Hv == 0 || Hv % Hk != 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (q_desc->strides()[3] != 1 || k_desc->strides()[3] != 1 || v_desc->strides()[3] != 1 || out_desc->strides()[3] != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        auto initial_shape = initial_state_desc->shape();
        size_t pool_size = initial_shape[0];
        if (indexed_pool) {
            // Indexed pool layout is [pool_size, Hv, Dv, Dk].
            if (initial_shape[1] != Hv || initial_shape[2] != Dv || initial_shape[3] != Dk) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        } else {
            // State layout is [B, Hv, Dv, Dk].
            if (initial_shape[0] != B || initial_shape[1] != Hv || initial_shape[2] != Dv || initial_shape[3] != Dk) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        if (!has_final_indices) {
            auto final_shape = final_state_desc->shape();
            if (indexed_pool) {
                if (final_shape[0] != B || final_shape[1] != Hv || final_shape[2] != Dv || final_shape[3] != Dk) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            } else {
                if (final_shape[0] != B || final_shape[1] != Hv || final_shape[2] != Dv || final_shape[3] != Dk) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            }
        }

        infiniDtype_t initial_indices_dtype = INFINI_DTYPE_INVALID;
        infiniDtype_t final_indices_dtype = INFINI_DTYPE_INVALID;
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

        RecurrentGatedDeltaRuleInfo info;
        info.data_dtype = data_dtype;
        info.gate_dtype = gate_dtype;
        info.initial_state_indices_dtype = initial_indices_dtype;
        info.final_state_indices_dtype = final_indices_dtype;
        info.use_qk_l2norm = use_qk_l2norm;
        info.has_initial_state_indices = has_initial_indices;
        info.has_final_state_indices = has_final_indices;
        info.indexed_state_pool = indexed_pool;
        info.B = B;
        info.Hk = Hk;
        info.Hv = Hv;
        info.T = T;
        info.Dk = Dk;
        info.Dv = Dv;
        info.pool_size = pool_size;
        info.value_heads_per_key_head = Hv / Hk;
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

        return utils::Result<RecurrentGatedDeltaRuleInfo>(info);
    }
};

} // namespace recurrent_gated_delta_rule
} // namespace op

#endif // __RECURRENT_GATED_DELTA_RULE_INFO_H__
