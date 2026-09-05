#include "../../handle.h"
#include "../../operator.h"
#include "infiniop/ops/grouped_gemm.h"

#ifdef ENABLE_CPU_API
#include "cpu/grouped_gemm_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/grouped_gemm_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/grouped_gemm_metax.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateGroupedGemmDescriptor(
    infiniopHandle_t handle,
    infiniopGroupedGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t group_sizes_desc) {

#define CREATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                   \
        return op::grouped_gemm::NAMESPACE::Descriptor::create(                  \
            handle,                                                              \
            reinterpret_cast<op::grouped_gemm::NAMESPACE::Descriptor **>(desc_ptr), \
            c_desc,                                                              \
            a_desc,                                                              \
            b_desc,                                                              \
            group_sizes_desc)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetGroupedGemmWorkspaceSize(
    infiniopGroupedGemmDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                              \
    case CASE:                                                                                            \
        *size = reinterpret_cast<const op::grouped_gemm::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__INFINI_C infiniStatus_t infiniopGroupedGemm(
    infiniopGroupedGemmDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    const void *group_sizes,
    const void *group_sizes_host,
    float alpha,
    float beta,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                         \
        return reinterpret_cast<const op::grouped_gemm::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size,                                     \
                        c, a, b, group_sizes, group_sizes_host,                        \
                        alpha, beta,                                                   \
                        stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyGroupedGemmDescriptor(
    infiniopGroupedGemmDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                   \
        delete reinterpret_cast<const op::grouped_gemm::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        DELETE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DELETE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
