#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/gptq_marlin_gemm.h"

#if defined ENABLE_NVIDIA_API
#include "nvidia/gptq_marlin_gemm_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateGptqMarlinGemmDescriptor(
    infiniopHandle_t handle,
    infiniopGptqMarlinGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t global_scale_desc,
    infiniopTensorDescriptor_t b_zeros_desc,
    infiniopTensorDescriptor_t g_idx_desc,
    infiniopTensorDescriptor_t perm_desc) {
#define CREATE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                          \
        return op::gptq_marlin_gemm::NAMESPACE::Descriptor::create(                     \
            handle,                                                                     \
            reinterpret_cast<op::gptq_marlin_gemm::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc,                                                                   \
            a_desc,                                                                     \
            b_desc,                                                                     \
            b_scales_desc,                                                              \
            global_scale_desc,                                                          \
            b_zeros_desc,                                                               \
            g_idx_desc,                                                                 \
            perm_desc)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetGptqMarlinGemmWorkspaceSize(infiniopGptqMarlinGemmDescriptor_t desc,
                                                                 size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                  \
    case CASE:                                                                                                \
        *size = reinterpret_cast<const op::gptq_marlin_gemm::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopGptqMarlinGemm(
    infiniopGptqMarlinGemmDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *a,
    const void *b,
    void *b_scales,
    void *global_scale,
    void *b_zeros,
    void *g_idx,
    void *perm,
    int64_t b_q_type_id,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                             \
        return reinterpret_cast<const op::gptq_marlin_gemm::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, out, a, b, b_scales, global_scale, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float, stream)

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
infiniopDestroyGptqMarlinGemmDescriptor(infiniopGptqMarlinGemmDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                             \
    case CASE:                                                                              \
        delete reinterpret_cast<const op::gptq_marlin_gemm::NAMESPACE::Descriptor *>(desc); \
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
