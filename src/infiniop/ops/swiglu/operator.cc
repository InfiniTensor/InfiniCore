#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/swiglu.h"

#ifdef ENABLE_CPU_API
#include "cpu/swiglu_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) 
#include "nvidia/swiglu_nvidia.cuh"
#endif
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#if defined(ENABLE_NINETOOTHED)
#include "ninetoothed/swiglu.h"
#else
#include "nvidia/swiglu_nvidia.cuh"
#endif
#endif
#ifdef ENABLE_KUNLUN_API
#include "kunlun/swiglu_kunlun.h"
#endif
#ifdef ENABLE_METAX_API
#if defined(ENABLE_NINETOOTHED)
#include "ninetoothed/swiglu.h"
#else
#include "metax/swiglu_metax.h"
#endif
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/swiglu_bang.h"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/swiglu_ascend.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/swiglu_moore.h"
#endif

__C infiniStatus_t infiniopCreateSwiGLUDescriptor(
    infiniopHandle_t handle,
    infiniopSwiGLUDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

#define CREATE(CASE, NAMESPACE)                                               \
    case CASE:                                                                \
        return op::swiglu::NAMESPACE::Descriptor::create(                     \
            handle,                                                           \
            reinterpret_cast<op::swiglu::NAMESPACE::Descriptor **>(desc_ptr), \
            c_desc,                                                           \
            {a_desc,                                                          \
             b_desc})

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
#ifdef ENABLE_NINETOOTHED
        CREATE(INFINI_DEVICE_ILUVATAR, ninetoothed);
#else
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_KUNLUN_API
        CREATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_METAX_API
#ifdef ENABLE_NINETOOTHED
        CREATE(INFINI_DEVICE_METAX, ninetoothed);
#else
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#endif
#ifdef ENABLE_CAMBRICON_API
        CREATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_ASCEND_API
        CREATE(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetSwiGLUWorkspaceSize(infiniopSwiGLUDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                  \
    case CASE:                                                                                \
        *size = reinterpret_cast<op::swiglu::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
#ifdef ENABLE_NINETOOTHED
        GET(INFINI_DEVICE_ILUVATAR, ninetoothed);
#else
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
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
#ifdef ENABLE_METAX_API
#ifdef ENABLE_NINETOOTHED
        GET(INFINI_DEVICE_METAX, ninetoothed);
#else
        GET(INFINI_DEVICE_METAX, metax);
#endif
#endif
#ifdef ENABLE_CAMBRICON_API
        GET(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_ASCEND_API
        GET(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore);
#endif
    }

#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopSwiGLU(
    infiniopSwiGLUDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                               \
    case CASE:                                                                   \
        return reinterpret_cast<const op::swiglu::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, c, {a, b}, stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
#ifdef ENABLE_NINETOOTHED
        CALCULATE(INFINI_DEVICE_ILUVATAR, ninetoothed);
#else
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_KUNLUN_API
        CALCULATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_METAX_API
#ifdef ENABLE_NINETOOTHED
        CALCULATE(INFINI_DEVICE_METAX, ninetoothed);
#else
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#endif
#ifdef ENABLE_CAMBRICON_API
        CALCULATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_ASCEND_API
        CALCULATE(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroySwiGLUDescriptor(infiniopSwiGLUDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                    \
        delete reinterpret_cast<const op::swiglu::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
#ifdef ENABLE_NINETOOTHED
        DELETE(INFINI_DEVICE_ILUVATAR, ninetoothed);
#else
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#endif
#ifdef ENABLE_ALI_API
        DELETE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_QY_API
        DELETE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DELETE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_KUNLUN_API
        DELETE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_METAX_API
#ifdef ENABLE_NINETOOTHED
        DELETE(INFINI_DEVICE_METAX, ninetoothed);
#else
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
#endif
#ifdef ENABLE_CAMBRICON_API
        DELETE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_ASCEND_API
        DELETE(INFINI_DEVICE_ASCEND, ascend)
#endif
#ifdef ENABLE_MOORE_API
        DELETE(INFINI_DEVICE_MOORE, moore);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
