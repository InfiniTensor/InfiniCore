#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/reduce_mean.h"

#ifdef ENABLE_CPU_API
#include "cpu/reduce_mean_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/reduce_mean_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/reduce_mean_metax.h"
#endif

__C infiniStatus_t infiniopCreateReduceMeanDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMeanDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t dim) {
    
    if (!handle || !desc_ptr || !output_desc || !input_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    switch (handle->device) {
#ifdef ENABLE_CPU_API
        case INFINI_DEVICE_CPU:
            return op::reduce_mean::cpu::Descriptor::create(
                handle, reinterpret_cast<op::reduce_mean::cpu::Descriptor**>(desc_ptr), output_desc, input_desc, dim);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
        case INFINI_DEVICE_NVIDIA:
        case INFINI_DEVICE_ILUVATAR:
            return op::reduce_mean::nvidia::Descriptor::create(
                handle, reinterpret_cast<op::reduce_mean::nvidia::Descriptor**>(desc_ptr), output_desc, input_desc, dim);
#endif
#ifdef ENABLE_METAX_API
        case INFINI_DEVICE_METAX:
            return op::reduce_mean::metax::Descriptor::create(
                handle, reinterpret_cast<op::reduce_mean::metax::Descriptor**>(desc_ptr), output_desc, input_desc, dim);
#endif
        default:
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopGetReduceMeanWorkspaceSize(
    infiniopReduceMeanDescriptor_t desc,
    size_t *size) {
    
    if (!desc || !size) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        case INFINI_DEVICE_CPU:
            *size = static_cast<const op::reduce_mean::cpu::Descriptor*>(desc)->workspaceSize();
            break;
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
        case INFINI_DEVICE_NVIDIA:
        case INFINI_DEVICE_ILUVATAR:
            *size = static_cast<const op::reduce_mean::nvidia::Descriptor*>(desc)->workspaceSize();
            break;
#endif
#ifdef ENABLE_METAX_API
        case INFINI_DEVICE_METAX:
            *size = static_cast<const op::reduce_mean::metax::Descriptor*>(desc)->workspaceSize();
            break;
#endif
        default:
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    
    return INFINI_STATUS_SUCCESS;
}

__C infiniStatus_t infiniopReduceMean(
    infiniopReduceMeanDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) {
    
    if (!desc || !output || !input) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // Note: workspace size check is handled by individual implementations
    // if needed, since base InfiniopDescriptor doesn't have workspace_size member
    
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        case INFINI_DEVICE_CPU:
            return static_cast<const op::reduce_mean::cpu::Descriptor*>(desc)->calculate(workspace, workspace_size, output, input, stream);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
        case INFINI_DEVICE_NVIDIA:
        case INFINI_DEVICE_ILUVATAR:
            return static_cast<const op::reduce_mean::nvidia::Descriptor*>(desc)->calculate(workspace, workspace_size, output, input, stream);
#endif
#ifdef ENABLE_METAX_API
        case INFINI_DEVICE_METAX:
            return static_cast<const op::reduce_mean::metax::Descriptor*>(desc)->calculate(workspace, workspace_size, output, input, stream);
#endif
        default:
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopDestroyReduceMeanDescriptor(
    infiniopReduceMeanDescriptor_t desc) {
    
    if (!desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    delete desc;
    return INFINI_STATUS_SUCCESS;
}