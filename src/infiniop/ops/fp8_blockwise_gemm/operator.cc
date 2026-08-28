#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/fp8_blockwise_gemm.h"

#ifdef ENABLE_CPU_API
#include "cpu/fp8_blockwise_gemm_cpu.h"
#endif
#ifdef ENABLE_NVIDIA_API
#include "nvidia/fp8_blockwise_gemm_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateFp8BlockwiseGemmDescriptor(
    infiniopHandle_t handle,
    infiniopFp8BlockwiseGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t scales_desc) {
#define CREATE(CASE, NAMESPACE)                                                              \
    case CASE:                                                                               \
        return op::fp8_blockwise_gemm::NAMESPACE::Descriptor::create(                        \
            handle,                                                                          \
            reinterpret_cast<op::fp8_blockwise_gemm::NAMESPACE::Descriptor **>(desc_ptr),    \
            out_desc, a_desc, q_desc, scales_desc)
    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetFp8BlockwiseGemmWorkspaceSize(
    infiniopFp8BlockwiseGemmDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                                            \
    case CASE:                                                                                          \
        *size = reinterpret_cast<const op::fp8_blockwise_gemm::NAMESPACE::Descriptor *>(desc)           \
                    ->workspaceSize();                                                                  \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopFp8BlockwiseGemm(
    infiniopFp8BlockwiseGemmDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *a,
    const void *q,
    const void *scales,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                                     \
    case CASE:                                                                                         \
        return reinterpret_cast<const op::fp8_blockwise_gemm::NAMESPACE::Descriptor *>(desc)           \
            ->calculate(workspace, workspace_size, out, a, q, scales, stream)
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyFp8BlockwiseGemmDescriptor(
    infiniopFp8BlockwiseGemmDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                                        \
    case CASE:                                                                                          \
        delete reinterpret_cast<const op::fp8_blockwise_gemm::NAMESPACE::Descriptor *>(desc);           \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
