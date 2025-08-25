#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/equal.h"

#ifdef ENABLE_CPU_API
#include "cpu/equal_cpu.h"
#endif

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/equal_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/equal_metax.h"
#endif
#ifdef ENABLE_KUNLUN_API
#include "kunlun/equal_kunlun.h"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/equal_bang.h"
#endif

__C infiniStatus_t infiniopCreateEqualDescriptor(
    infiniopHandle_t handle,
    infiniopEqualDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

#define CREATE(CASE, NAMESPACE)                                              \
    case CASE:                                                               \
        return op::equal::NAMESPACE::Descriptor::create(                     \
            handle,                                                          \
            reinterpret_cast<op::equal::NAMESPACE::Descriptor **>(desc_ptr), \
            c_desc,                                                          \
            {a_desc,                                                         \
             b_desc})

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
        return INFINI_STATUS_BAD_PARAM;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetEqualWorkspaceSize(infiniopEqualDescriptor_t desc, size_t *size) {

#define GET_WORKSPACE_SIZE(CASE, NAMESPACE)                                \
    case CASE:                                                             \
        *size = reinterpret_cast<op::equal::NAMESPACE::Descriptor *>(desc) \
            ->workspaceSize();                                             \
        return INFINI_STATUS_SUCCESS

    switch (reinterpret_cast<InfiniopDescriptor *>(desc)->device_type) {

#ifdef ENABLE_CPU_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_KUNLUN_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_CAMBRICON_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_CAMBRICON, bang);
#endif

    default:
        return INFINI_STATUS_BAD_PARAM;
    }

#undef GET_WORKSPACE_SIZE
}

__C infiniStatus_t infiniopEqual(
    infiniopEqualDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                         \
    case CASE:                                                             \
        return reinterpret_cast<op::equal::NAMESPACE::Descriptor *>(desc)  \
            ->calculate(workspace, workspace_size, c, {a, b}, stream)

    switch (reinterpret_cast<InfiniopDescriptor *>(desc)->device_type) {

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
        return INFINI_STATUS_BAD_PARAM;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyEqualDescriptor(infiniopEqualDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                           \
    case CASE:                                                             \
        delete reinterpret_cast<op::equal::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (reinterpret_cast<InfiniopDescriptor *>(desc)->device_type) {

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
        return INFINI_STATUS_BAD_PARAM;
    }

#undef DESTROY
}