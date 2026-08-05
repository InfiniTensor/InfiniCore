#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/mamba_selective_scan.h"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_ALI_API) || defined(ENABLE_HYGON_API)
#include "nvidia/mamba_selective_scan_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/mamba_selective_scan_metax.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateMambaSelectiveScanDescriptor(
    infiniopHandle_t handle, infiniopMambaSelectiveScanDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc, infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t dt_desc, infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_log_desc,
    infiniopTensorDescriptor_t d_desc, infiniopTensorDescriptor_t gate_desc,
    infiniopTensorDescriptor_t dt_bias_desc, infiniopTensorDescriptor_t state_desc) {
#define CREATE(CASE, NAMESPACE) \
    case CASE:                  \
        return op::mamba_selective_scan::NAMESPACE::Descriptor::create(handle, reinterpret_cast<op::mamba_selective_scan::NAMESPACE::Descriptor **>(desc_ptr), out_desc, x_desc, dt_desc, b_desc, c_desc, a_log_desc, d_desc, gate_desc, dt_bias_desc, state_desc)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
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

__INFINI_C infiniStatus_t infiniopGetMambaSelectiveScanWorkspaceSize(infiniopMambaSelectiveScanDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                \
    case CASE:                                                                                              \
        *size = reinterpret_cast<op::mamba_selective_scan::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
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

__INFINI_C infiniStatus_t infiniopMambaSelectiveScan(
    infiniopMambaSelectiveScanDescriptor_t desc, void *workspace, size_t workspace_size,
    void *out, const void *x, const void *dt, const void *b, const void *c,
    const void *a_log, const void *d, const void *gate, const void *dt_bias,
    void *state, void *stream) {
#define CALC(CASE, NAMESPACE) \
    case CASE:                \
        return reinterpret_cast<op::mamba_selective_scan::NAMESPACE::Descriptor *>(desc)->calculate(workspace, workspace_size, out, x, dt, b, c, a_log, d, gate, dt_bias, state, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALC(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALC(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALC(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALC(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALC(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALC(INFINI_DEVICE_METAX, metax);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALC
}

__INFINI_C infiniStatus_t infiniopDestroyMambaSelectiveScanDescriptor(infiniopMambaSelectiveScanDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                          \
    case CASE:                                                                            \
        delete reinterpret_cast<op::mamba_selective_scan::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
