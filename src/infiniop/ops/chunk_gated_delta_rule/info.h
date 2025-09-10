// infiniop/ops/chunk_gated_delta_rule/info.h

#ifndef __CHUNK_GATED_DELTA_RULE_INFO_H__
#define __CHUNK_GATED_DELTA_RULE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>
#include <optional>

namespace op {
namespace chunk_gated_delta_rule {

class ChunkGatedDeltaRuleInfo {
    ChunkGatedDeltaRuleInfo() = default;

public:
    // --- Data Types and Flags ---
    infiniDtype_t dtype;
    bool use_qk_l2norm;

    // --- Shape Dimensions ---
    size_t B, H, T, Dk, Dv, chunk_size;

    // --- Strides for Memory Layout ---
    // Strides can be added here if needed for more complex layouts

    static utils::Result<ChunkGatedDeltaRuleInfo>
    create(infiniopTensorDescriptor_t out_desc,
           infiniopTensorDescriptor_t final_state_desc,
           infiniopTensorDescriptor_t q_desc,
           infiniopTensorDescriptor_t k_desc,
           infiniopTensorDescriptor_t v_desc,
           infiniopTensorDescriptor_t g_desc,
           infiniopTensorDescriptor_t beta_desc,
           infiniopTensorDescriptor_t initial_state_desc,
           bool use_qk_l2norm,
           size_t chunk_size) {
        
        auto dtype = q_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        
        // Check for consistent data types across all tensors
        if (out_desc->dtype() != dtype || final_state_desc->dtype() != dtype ||
            k_desc->dtype() != dtype || v_desc->dtype() != dtype ||
            g_desc->dtype() != dtype || beta_desc->dtype() != dtype ||
            initial_state_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // Check tensor dimensions
        if (q_desc->ndim() != 4 || k_desc->ndim() != 4 || v_desc->ndim() != 4 ||
            g_desc->ndim() != 3 || beta_desc->ndim() != 3 ||
            initial_state_desc->ndim() != 4 || out_desc->ndim() != 4 ||
            final_state_desc->ndim() != 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        
        ChunkGatedDeltaRuleInfo info;
        info.dtype = dtype;
        info.use_qk_l2norm = use_qk_l2norm;
        info.chunk_size = chunk_size;
        
        auto q_shape = q_desc->shape();
        info.B = q_shape[0];
        info.H = q_shape[1];
        info.T = q_shape[2];
        info.Dk = q_shape[3];
        
        info.Dv = v_desc->shape()[3];

        // Further validation can be added here to ensure all shapes are compatible.
        // For example, check if initial_state_desc shape is [B, H, Dk, Dv].
        
        return utils::Result<ChunkGatedDeltaRuleInfo>(info);
    }
};

} // namespace chunk_gated_delta_rule
} // namespace op

#endif // __CHUNK_GATED_DELTA_RULE_INFO_H__