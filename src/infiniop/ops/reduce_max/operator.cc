#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/reduce_max.h"

#ifdef ENABLE_CPU_API
#include "cpu/reduce_max_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/reduce_max_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/reduce_max_metax.h"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/reduce_max_ascend.h"
#endif

__C infiniStatus_t infiniopCreateReduceMaxDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t dim) {

#define CREATE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                    \
        return op::reduce_max::NAMESPACE::Descriptor::create(                     \
            handle,                                                               \
            reinterpret_cast<op::reduce_max::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc,                                                          \
            input_desc,                                                           \
            dim);

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax)
#endif
        // #ifdef ENABLE_ASCEND_API
        //         CREATE(INFINI_DEVICE_ASCEND, ascend)
        // #endif
        // #ifdef ENABLE_CAMBRICON_MLU
        //     case DevCambriconMlu: {
        //         return bangCreateReduceMaxDescriptor((BangHandle_t)handle, (ReduceMaxBangDescriptor_t *)desc_ptr, output_desc, input_desc, dim);
        //         // return cnnlCreateReduceMaxDescriptor((BangHandle_t) handle, (ReduceMaxCnnlDescriptor_t *) desc_ptr, output_desc, input_desc, dim);
        //     }
        // #endif
        // #ifdef ENABLE_MTHREADS_GPU
        //     case DevMthreadsGpu: {
        //         return musaCreateReduceMaxDescriptor((MusaHandle_t)handle, (ReduceMaxMusaDescriptor_t *)desc_ptr, output_desc, input_desc, dim);
        //     }
        // #endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetReduceMaxWorkspaceSize(infiniopReduceMaxDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                    \
        *size = reinterpret_cast<op::reduce_max::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax)
#endif
        // #ifdef ENABLE_ASCEND_API
        //         GET(INFINI_DEVICE_ASCEND, ascend)
        // #endif
        // #ifdef ENABLE_CAMBRICON_MLU
        //     case DevCambriconMlu: {
        //         return bangGetReduceMaxWorkspaceSize((ReduceMaxBangDescriptor_t)desc, size);
        //         // return cnnlGetReduceMaxWorkspaceSize((ReduceMaxCnnlDescriptor_t) desc, size);
        //     }
        // #endif
        // #ifdef ENABLE_MTHREADS_GPU
        //     case DevMthreadsGpu: {
        //         return musaGetReduceMaxWorkspaceSize((ReduceMaxMusaDescriptor_t)desc, size);
        //     }
        // #endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopReduceMax(
    infiniopReduceMaxDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *output,
    const void *input,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                             \
        return reinterpret_cast<op::reduce_max::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, output, input, stream);

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax)
#endif
        // #ifdef ENABLE_ASCEND_API
        //         CALCULATE(INFINI_DEVICE_ASCEND, ascend)
        // #endif
        // #ifdef ENABLE_CAMBRICON_MLU
        //     case DevCambriconMlu: {
        //         return bangReduceMax((ReduceMaxBangDescriptor_t)desc, workspace, workspace_size, output, input, stream);
        //         // return cnnlReduceMax((ReduceMaxCnnlDescriptor_t) desc, workspace, workspace_size, output, input, stream);
        //     }
        // #endif
        // #ifdef ENABLE_MTHREADS_GPU
        //     case DevMthreadsGpu: {
        //         return musaReduceMax((ReduceMaxMusaDescriptor_t)desc, workspace, workspace_size, output, input, stream);
        //     }
        // #endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyReduceMaxDescriptor(infiniopReduceMaxDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                \
    case CASE:                                                                  \
        delete reinterpret_cast<op::reduce_max::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax)
#endif
        // #ifdef ENABLE_ASCEND_API
        //         DESTROY(INFINI_DEVICE_ASCEND, ascend)
        // #endif
        // #ifdef ENABLE_CAMBRICON_MLU
        //     case DevCambriconMlu: {
        //         return bangDestroyReduceMaxDescriptor((ReduceMaxBangDescriptor_t)desc);
        //         // return cnnlDestroyReduceMaxDescriptor((ReduceMaxCnnlDescriptor_t) desc);
        //     }
        // #endif
        // #ifdef ENABLE_MTHREADS_GPU
        //     case DevMthreadsGpu:
        //         return musaDestroyReduceMaxDescriptor((ReduceMaxMusaDescriptor_t)desc);
        // #endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}