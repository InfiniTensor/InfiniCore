#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/layer_norm.h"

#ifdef ENABLE_CPU_API
#include "cpu/layer_norm_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/layer_norm_nvidia.cuh"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/layer_norm_ascend.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/layer_norm_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "musa/layer_norm_musa.cuh"
#endif
#ifdef ENABLE_KUNLUN_API
#include "kunlun/layer_norm_kunlun.h"
#endif

__C infiniStatus_t infiniopCreateLayerNormDescriptor(
    infiniopHandle_t handle,
    infiniopLayerNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    infiniopTensorDescriptor_t input_std_deviation_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    float eps) {
    
    if (!handle || !desc_ptr || !output_desc || !input_desc || 
        !weight_desc || !input_std_deviation_desc || !input_standardization_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
#define CREATE(CASE, NAMESPACE) \
    case CASE: \
        return op::layer_norm::NAMESPACE::Descriptor::create( \
            handle, reinterpret_cast<op::layer_norm::NAMESPACE::Descriptor**>(desc_ptr), \
            output_desc, input_desc, weight_desc, bias_desc, \
            input_std_deviation_desc, input_standardization_desc, eps)

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
#ifdef ENABLE_ASCEND_API
        case INFINI_DEVICE_ASCEND: {
            op::layer_norm::ascend::Descriptor *desc;
            CHECK_STATUS(op::layer_norm::ascend::Descriptor::create(
                handle, &desc, output_desc, input_desc, weight_desc, bias_desc,
                input_std_deviation_desc, input_standardization_desc, eps));
            *desc_ptr = reinterpret_cast<infiniopLayerNormDescriptor_t>(desc);
            return INFINI_STATUS_SUCCESS;
        }
#endif
#ifdef ENABLE_METAX_API
        case INFINI_DEVICE_METAX: {
            op::layer_norm::metax::Descriptor *desc;
            CHECK_STATUS(op::layer_norm::metax::Descriptor::create(
                handle, &desc, output_desc, input_desc, weight_desc, bias_desc,
                input_std_deviation_desc, input_standardization_desc, eps));
            *desc_ptr = reinterpret_cast<infiniopLayerNormDescriptor_t>(desc);
            return INFINI_STATUS_SUCCESS;
        }
#endif
#ifdef ENABLE_MOORE_API
        case INFINI_DEVICE_MOORE: {
            op::layer_norm::musa::Descriptor *desc;
            CHECK_STATUS(op::layer_norm::musa::Descriptor::create(
                handle, &desc, output_desc, input_desc, weight_desc, bias_desc,
                input_std_deviation_desc, input_standardization_desc, eps));
            *desc_ptr = reinterpret_cast<infiniopLayerNormDescriptor_t>(desc);
            return INFINI_STATUS_SUCCESS;
        }
#endif
#ifdef ENABLE_KUNLUN_API
        case INFINI_DEVICE_KUNLUN: {
            op::layer_norm::kunlun::Descriptor *desc;
            CHECK_STATUS(op::layer_norm::kunlun::Descriptor::create(
                handle, &desc, output_desc, input_desc, weight_desc, bias_desc,
                input_std_deviation_desc, input_standardization_desc, eps));
            *desc_ptr = reinterpret_cast<infiniopLayerNormDescriptor_t>(desc);
            return INFINI_STATUS_SUCCESS;
        }
#endif
    }

#undef CREATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetLayerNormWorkspaceSize(
    infiniopLayerNormDescriptor_t desc,
    size_t *size) {
    
    if (!desc || !size) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
#define GET(CASE, NAMESPACE) \
    case CASE: \
        return reinterpret_cast<op::layer_norm::NAMESPACE::Descriptor*>(desc)->get_workspace_size(size)

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
#ifdef ENABLE_ASCEND_API
        case INFINI_DEVICE_ASCEND:
            return reinterpret_cast<op::layer_norm::ascend::Descriptor *>(desc)->get_workspace_size(size);
#endif
#ifdef ENABLE_METAX_API
        case INFINI_DEVICE_METAX:
            return reinterpret_cast<op::layer_norm::metax::Descriptor *>(desc)->get_workspace_size(size);
#endif
#ifdef ENABLE_MOORE_API
        case INFINI_DEVICE_MOORE:
            return reinterpret_cast<op::layer_norm::musa::Descriptor *>(desc)->get_workspace_size(size);
#endif
#ifdef ENABLE_KUNLUN_API
        case INFINI_DEVICE_KUNLUN:
            return reinterpret_cast<op::layer_norm::kunlun::Descriptor *>(desc)->get_workspace_size(size);
#endif
    }

#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopLayerNorm(
    infiniopLayerNormDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *weight,
    const void *bias,
    void *input_std_deviation,
    void *input_standardization,
    void *stream) {
    
    if (!desc || !output || !input || !weight || !input_std_deviation || !input_standardization) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
#define CALCULATE(CASE, NAMESPACE) \
    case CASE: \
        return reinterpret_cast<op::layer_norm::NAMESPACE::Descriptor*>(desc)->calculate( \
            workspace, workspace_size, output, input, weight, bias, \
            input_std_deviation, input_standardization, stream)

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
#ifdef ENABLE_ASCEND_API
        case INFINI_DEVICE_ASCEND:
            return reinterpret_cast<op::layer_norm::ascend::Descriptor *>(desc)->calculate(
                workspace, workspace_size, output, input, weight, bias,
                input_std_deviation, input_standardization, stream);
#endif
#ifdef ENABLE_METAX_API
        case INFINI_DEVICE_METAX:
            return reinterpret_cast<op::layer_norm::metax::Descriptor *>(desc)->calculate(
                workspace, workspace_size, output, input, weight, bias,
                input_std_deviation, input_standardization, stream);
#endif
#ifdef ENABLE_MOORE_API
        case INFINI_DEVICE_MOORE:
            return reinterpret_cast<op::layer_norm::musa::Descriptor *>(desc)->calculate(
                workspace, workspace_size, output, input, weight, bias,
                input_std_deviation, input_standardization, stream);
#endif
#ifdef ENABLE_KUNLUN_API
        case INFINI_DEVICE_KUNLUN:
            return reinterpret_cast<op::layer_norm::kunlun::Descriptor *>(desc)->calculate(
                workspace, workspace_size, output, input, weight, bias,
                input_std_deviation, input_standardization, stream);
#endif
    }

#undef CALCULATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyLayerNormDescriptor(
    infiniopLayerNormDescriptor_t desc) {
    
    if (!desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
#define DESTROY(CASE, NAMESPACE) \
    case CASE: \
        delete reinterpret_cast<op::layer_norm::NAMESPACE::Descriptor*>(desc); \
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
#ifdef ENABLE_ASCEND_API
        case INFINI_DEVICE_ASCEND:
            delete reinterpret_cast<op::layer_norm::ascend::Descriptor *>(desc);
            return INFINI_STATUS_SUCCESS;
#endif
#ifdef ENABLE_METAX_API
        case INFINI_DEVICE_METAX:
            delete reinterpret_cast<op::layer_norm::metax::Descriptor *>(desc);
            return INFINI_STATUS_SUCCESS;
#endif
#ifdef ENABLE_MOORE_API
        case INFINI_DEVICE_MOORE:
            delete reinterpret_cast<op::layer_norm::musa::Descriptor *>(desc);
            return INFINI_STATUS_SUCCESS;
#endif
#ifdef ENABLE_KUNLUN_API
        case INFINI_DEVICE_KUNLUN:
            delete reinterpret_cast<op::layer_norm::kunlun::Descriptor *>(desc);
            return INFINI_STATUS_SUCCESS;
#endif
    }

#undef DESTROY

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}