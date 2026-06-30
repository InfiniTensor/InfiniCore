#include "../../operator.h"
#include "../../handle.h"

#include "infiniop/ops/awq_marlin_gemm.h"
#include "infiniop/ops/gptq_marlin_gemm.h"

namespace {

infiniStatus_t marlinNotEnabled() {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

} // namespace

__INFINI_C infiniStatus_t infiniopCreateAwqMarlinGemmDescriptor(
    infiniopHandle_t handle,
    infiniopAwqMarlinGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_bias_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t a_scales_desc,
    infiniopTensorDescriptor_t global_scales_desc,
    infiniopTensorDescriptor_t b_zeros_desc,
    infiniopTensorDescriptor_t g_idx_desc,
    infiniopTensorDescriptor_t perm_desc) {
    (void)handle;
    (void)desc_ptr;
    (void)out_desc;
    (void)a_desc;
    (void)b_desc;
    (void)b_bias_desc;
    (void)b_scales_desc;
    (void)a_scales_desc;
    (void)global_scales_desc;
    (void)b_zeros_desc;
    (void)g_idx_desc;
    (void)perm_desc;
    return marlinNotEnabled();
}

__INFINI_C infiniStatus_t infiniopGetAwqMarlinGemmWorkspaceSize(
    infiniopAwqMarlinGemmDescriptor_t desc,
    size_t *size) {
    (void)desc;
    (void)size;
    return marlinNotEnabled();
}

__INFINI_C infiniStatus_t infiniopAwqMarlinGemm(
    infiniopAwqMarlinGemmDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    void *b_bias,
    void *b_scales,
    void *a_scales,
    void *global_scales,
    void *b_zeros,
    void *g_idx,
    void *perm,
    int64_t b_q_type_id,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float,
    void *stream) {
    (void)desc;
    (void)workspace;
    (void)workspace_size;
    (void)c;
    (void)a;
    (void)b;
    (void)b_bias;
    (void)b_scales;
    (void)a_scales;
    (void)global_scales;
    (void)b_zeros;
    (void)g_idx;
    (void)perm;
    (void)b_q_type_id;
    (void)is_k_full;
    (void)use_atomic_add;
    (void)use_fp32_reduce;
    (void)is_zp_float;
    (void)stream;
    return marlinNotEnabled();
}

__INFINI_C infiniStatus_t infiniopDestroyAwqMarlinGemmDescriptor(
    infiniopAwqMarlinGemmDescriptor_t desc) {
    (void)desc;
    return marlinNotEnabled();
}

__INFINI_C infiniStatus_t infiniopCreateGptqMarlinGemmDescriptor(
    infiniopHandle_t handle,
    infiniopGptqMarlinGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t global_scales_desc,
    infiniopTensorDescriptor_t b_zeros_desc,
    infiniopTensorDescriptor_t g_idx_desc,
    infiniopTensorDescriptor_t perm_desc) {
    (void)handle;
    (void)desc_ptr;
    (void)out_desc;
    (void)a_desc;
    (void)b_desc;
    (void)b_scales_desc;
    (void)global_scales_desc;
    (void)b_zeros_desc;
    (void)g_idx_desc;
    (void)perm_desc;
    return marlinNotEnabled();
}

__INFINI_C infiniStatus_t infiniopGetGptqMarlinGemmWorkspaceSize(
    infiniopGptqMarlinGemmDescriptor_t desc,
    size_t *size) {
    (void)desc;
    (void)size;
    return marlinNotEnabled();
}

__INFINI_C infiniStatus_t infiniopGptqMarlinGemm(
    infiniopGptqMarlinGemmDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *a,
    const void *b,
    void *b_scales,
    void *global_scales,
    void *b_zeros,
    void *g_idx,
    void *perm,
    int64_t b_q_type_id,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float,
    void *stream) {
    (void)desc;
    (void)workspace;
    (void)workspace_size;
    (void)out;
    (void)a;
    (void)b;
    (void)b_scales;
    (void)global_scales;
    (void)b_zeros;
    (void)g_idx;
    (void)perm;
    (void)b_q_type_id;
    (void)is_k_full;
    (void)use_atomic_add;
    (void)use_fp32_reduce;
    (void)is_zp_float;
    (void)stream;
    return marlinNotEnabled();
}

__INFINI_C infiniStatus_t infiniopDestroyGptqMarlinGemmDescriptor(
    infiniopGptqMarlinGemmDescriptor_t desc) {
    (void)desc;
    return marlinNotEnabled();
}
