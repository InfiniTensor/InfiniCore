#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/add_rms_norm.h"

#ifdef ENABLE_CPU_API
#include "cpu/add_rms_norm_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
#include "nvidia/add_rms_norm_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/add_rms_norm_metax.cuh"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/add_rms_norm_moore.h"
#endif

__C infiniStatus_t infiniopCreateAddRMSNormDescriptor(
    infiniopHandle_t handle,
    infiniopAddRMSNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t x2_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {

#define CREATE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                      \
        return op::add_rms_norm::NAMESPACE::Descriptor::create(                     \
            handle,                                                                 \
            reinterpret_cast<op::add_rms_norm::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                                 \
            x1_desc,                                                                \
            x2_desc,                                                                \
            w_desc,                                                                 \
            epsilon)

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
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
#endif
    }

#undef CREATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetAddRMSNormWorkspaceSize(infiniopAddRMSNormDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                        \
    case CASE:                                                                                      \
        *size = reinterpret_cast<op::add_rms_norm::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_KUNLUN_API
        GET(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_CAMBRICON_API
        GET(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_ASCEND_API
        GET(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore);
#endif
    }

#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopAddRMSNorm(infiniopAddRMSNormDescriptor_t desc, void *workspace, size_t workspace_size,
                                      void *y, const void *x1, const void *x2, const void *w, void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                           \
    case CASE:                                                                               \
        return reinterpret_cast<op::add_rms_norm::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, y, x1, x2, w, stream)

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
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
#endif
    }

#undef CALCULATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyAddRMSNormDescriptor(infiniopAddRMSNormDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                  \
    case CASE:                                                                    \
        delete reinterpret_cast<op::add_rms_norm::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_KUNLUN_API
        DESTROY(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_CAMBRICON_API
        DESTROY(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_ASCEND_API
        DESTROY(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        DESTROY(INFINI_DEVICE_MOORE, moore);
#endif
    }

#undef DESTROY

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
