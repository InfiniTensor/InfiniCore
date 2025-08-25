#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/linear.h"

#ifdef ENABLE_CPU_API
#include "cpu/linear_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/linear_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/linear_metax.h"
#endif
#ifdef ENABLE_KUNLUN_API
#include "kunlun/linear_kunlun.h"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/linear_bang.h"
#endif

__C infiniStatus_t infiniopCreateLinearDescriptor(
    infiniopHandle_t handle,
    infiniopLinearDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t y_desc) {

#define CREATE(CASE, NAMESPACE)                                            \
    case CASE:                                                             \
        return op::linear::NAMESPACE::Descriptor::create(                  \
            handle,                                                        \
            reinterpret_cast<op::linear::NAMESPACE::Descriptor **>(desc_ptr), \
            x_desc,                                                        \
            w_desc,                                                        \
            b_desc,                                                        \
            y_desc)

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
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_KUNLUN_API
        CREATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_CAMBRICON_API
        CREATE(INFINI_DEVICE_CAMBRICON, bang);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetLinearWorkspaceSize(infiniopLinearDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                               \
    case CASE:                                                                             \
        *size = reinterpret_cast<op::linear::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_KUNLUN_API
        GET(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_CAMBRICON_API
        GET(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopLinear(
    infiniopLinearDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *w,
    const void *b,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                         \
        return reinterpret_cast<op::linear::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, y, x, w, b, stream)

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
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_KUNLUN_API
        CALCULATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_CAMBRICON_API
        CALCULATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyLinearDescriptor(infiniopLinearDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                           \
    case CASE:                                                             \
        delete reinterpret_cast<op::linear::NAMESPACE::Descriptor *>(desc); \
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
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_KUNLUN_API
        DESTROY(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_CAMBRICON_API
        DESTROY(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}