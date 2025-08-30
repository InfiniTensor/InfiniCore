#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/leakyrelu.h"

#ifdef ENABLE_CPU_API
#include "cpu/leakyrelu_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/leakyrelu_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/leakyrelu_metax.h"
#endif

__C infiniStatus_t infiniopCreateLeakyreluDescriptor(
    infiniopHandle_t handle,
    infiniopLeakyreluDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    float negative_slope) {

#define CREATE_LEAKY(CASE, NAMESPACE)                                            \
    case CASE:                                                                   \
        return op::leakyrelu::NAMESPACE::Descriptor::create(                     \
            handle,                                                              \
            reinterpret_cast<op::leakyrelu::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                              \
            x_desc,                                                              \
            negative_slope)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE_LEAKY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE_LEAKY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE_LEAKY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_KUNLUN_API
        CREATE_LEAKY(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_ASCEND_API
        CREATE_LEAKY(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_API
        CREATE_LEAKY(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CREATE_LEAKY(INFINI_DEVICE_MOORE, musa);
#endif
    }

#undef CREATE_LEAKY
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetLeakyreluWorkspaceSize(infiniopLeakyreluDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                     \
    case CASE:                                                                                   \
        *size = reinterpret_cast<op::leakyrelu::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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
#ifdef ENABLE_KUNLUN_API
        GET(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_ASCEND_API
        GET(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, musa);
#endif
    }

#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopLeakyrelu(infiniopLeakyreluDescriptor_t desc, void *workspace, size_t workspace_size,
                                     void *y, const void *x, void *stream) {

#define CALC_LEAKY(CASE, NAMESPACE)                                                       \
    case CASE:                                                                            \
        return reinterpret_cast<op::leakyrelu::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, y, x, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALC_LEAKY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALC_LEAKY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALC_LEAKY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_KUNLUN_API
        CALC_LEAKY(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_ASCEND_API
        CALC_LEAKY(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_API
        CALC_LEAKY(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CALC_LEAKY(INFINI_DEVICE_MOORE, musa);
#endif
    }

#undef CALC_LEAKY
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyLeakyreluDescriptor(infiniopLeakyreluDescriptor_t desc) {

#define DESTROY_LEAKY(CASE, NAMESPACE)                                         \
    case CASE:                                                                 \
        delete reinterpret_cast<op::leakyrelu::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY_LEAKY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY_LEAKY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY_LEAKY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_KUNLUN_API
        DESTROY_LEAKY(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_ASCEND_API
        DESTROY_LEAKY(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_API
        DESTROY_LEAKY(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        DESTROY_LEAKY(INFINI_DEVICE_MOORE, musa);
#endif
    }

#undef DESTROY_LEAKY
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
