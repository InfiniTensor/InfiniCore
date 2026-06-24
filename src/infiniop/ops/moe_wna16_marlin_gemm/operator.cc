#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/moe_wna16_marlin_gemm.h"

#if defined ENABLE_NVIDIA_API
#include "nvidia/moe_wna16_marlin_gemm_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateMoeWna16MarlinGemmDescriptor(
    infiniopHandle_t handle,
    infiniopMoeWna16MarlinGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_q_weight_desc,
    infiniopTensorDescriptor_t b_bias_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t global_scales_desc,
    infiniopTensorDescriptor_t b_zeros_desc,
    infiniopTensorDescriptor_t g_idx_desc,
    infiniopTensorDescriptor_t perm_desc,
    infiniopTensorDescriptor_t sorted_token_desc,
    infiniopTensorDescriptor_t expert_ids_desc,
    infiniopTensorDescriptor_t num_tokens_post_padded_desc,
    infiniopTensorDescriptor_t topk_weights_desc, 
    int size_m, int size_n, int size_k,
    int top_k, int moe_block_size) {
#define CREATE(CASE, NAMESPACE)                                                              \
    case CASE:                                                                               \
        return op::moe_wna16_marlin_gemm::NAMESPACE::Descriptor::create(                     \
            handle,                                                                          \
            reinterpret_cast<op::moe_wna16_marlin_gemm::NAMESPACE::Descriptor **>(desc_ptr), \
            c_desc,                                                                          \
            a_desc,                                                                          \
            b_q_weight_desc,                                                                 \
            b_bias_desc,                                                                     \
            b_scales_desc,                                                                   \
            global_scales_desc,                                                              \
            b_zeros_desc,                                                                    \
            g_idx_desc,                                                                      \
            perm_desc,                                                                       \
            sorted_token_desc,                                                               \
            expert_ids_desc,                                                                 \
            num_tokens_post_padded_desc,                                                     \
            topk_weights_desc,                                                               \
            size_m,                                                                          \
            size_n,                                                                          \
            size_k,                                                                          \
            top_k,                                                                           \
            moe_block_size)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetMoeWna16MarlinGemmWorkspaceSize(infiniopMoeWna16MarlinGemmDescriptor_t desc,
                                                                     size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                       \
    case CASE:                                                                                                     \
        *size = reinterpret_cast<const op::moe_wna16_marlin_gemm::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopMoeWna16MarlinGemm(
    infiniopMoeWna16MarlinGemmDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b_q_weight,
    void *b_bias,
    void *b_scales,
    void *global_scales,
    void *b_zeros,
    void *g_idx,
    void *perm,
    void *sorted_token_ids,
    void *expert_ids,
    void *num_tokens_post_padded,
    void *topk_weights,
    bool mul_topk_weights,
    bool is_ep,
    int64_t b_q_type_id,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                              \
    case CASE:                                                                                  \
        return reinterpret_cast<const op::moe_wna16_marlin_gemm::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, c, a, b_q_weight, b_bias, b_scales, global_scales, b_zeros, g_idx, perm, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, mul_topk_weights, is_ep, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t
infiniopDestroyMoeWna16MarlinGemmDescriptor(infiniopMoeWna16MarlinGemmDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                                  \
    case CASE:                                                                                   \
        delete reinterpret_cast<const op::moe_wna16_marlin_gemm::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}

// #endif
