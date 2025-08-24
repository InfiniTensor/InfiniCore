#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/batch_norm_backward.h"

#ifdef ENABLE_CPU_API
#include "cpu/batch_norm_backward_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/batch_norm_backward_nvidia.cuh"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/batch_norm_backward_ascend.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/batch_norm_backward_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "musa/batch_norm_backward_musa.cuh"
#endif
#ifdef ENABLE_KUNLUN_API
#include "kunlun/batch_norm_backward_kunlun.h"
#endif

__C infiniStatus_t infiniopCreateBatchNormBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopBatchNormBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t input_grad_desc,
    infiniopTensorDescriptor_t weight_grad_desc,
    infiniopTensorDescriptor_t bias_grad_desc,
    infiniopTensorDescriptor_t output_grad_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t running_mean_desc,
    infiniopTensorDescriptor_t running_var_desc,
    float momentum,
    float eps) {
    
    if (!handle || !desc_ptr || !input_grad_desc || !output_grad_desc || !input_desc || 
        !weight_desc || !running_mean_desc || !running_var_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
#define CREATE(CASE, NAMESPACE) \
    case CASE: \
        return op::batch_norm_backward::NAMESPACE::Descriptor::create( \
            handle, reinterpret_cast<op::batch_norm_backward::NAMESPACE::Descriptor**>(desc_ptr), \
            input_grad_desc, weight_grad_desc, bias_grad_desc, output_grad_desc, \
            input_desc, weight_desc, running_mean_desc, running_var_desc, momentum, eps)

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
            op::batch_norm_backward::ascend::Descriptor *desc;
            CHECK_STATUS(op::batch_norm_backward::ascend::Descriptor::create(
                handle, &desc, input_grad_desc, weight_grad_desc, bias_grad_desc, output_grad_desc,
                input_desc, weight_desc, running_mean_desc, running_var_desc, momentum, eps));
            *desc_ptr = reinterpret_cast<infiniopBatchNormBackwardDescriptor_t>(desc);
            return INFINI_STATUS_SUCCESS;
        }
#endif
#ifdef ENABLE_METAX_API
        case INFINI_DEVICE_METAX: {
            op::batch_norm_backward::metax::Descriptor *desc;
            CHECK_STATUS(op::batch_norm_backward::metax::Descriptor::create(
                handle, &desc, input_grad_desc, weight_grad_desc, bias_grad_desc, output_grad_desc,
                input_desc, weight_desc, running_mean_desc, running_var_desc, momentum, eps));
            *desc_ptr = reinterpret_cast<infiniopBatchNormBackwardDescriptor_t>(desc);
            return INFINI_STATUS_SUCCESS;
        }
#endif
#ifdef ENABLE_MOORE_API
        case INFINI_DEVICE_MOORE: {
            op::batch_norm_backward::musa::Descriptor *desc;
            CHECK_STATUS(op::batch_norm_backward::musa::Descriptor::create(
                handle, &desc, input_grad_desc, weight_grad_desc, bias_grad_desc, output_grad_desc,
                input_desc, weight_desc, running_mean_desc, running_var_desc, momentum, eps));
            *desc_ptr = reinterpret_cast<infiniopBatchNormBackwardDescriptor_t>(desc);
            return INFINI_STATUS_SUCCESS;
        }
#endif
#ifdef ENABLE_KUNLUN_API
        case INFINI_DEVICE_KUNLUN: {
            op::batch_norm_backward::kunlun::Descriptor *desc;
            CHECK_STATUS(op::batch_norm_backward::kunlun::Descriptor::create(
                handle, &desc, input_grad_desc, weight_grad_desc, bias_grad_desc, output_grad_desc,
                input_desc, weight_desc, running_mean_desc, running_var_desc, momentum, eps));
            *desc_ptr = reinterpret_cast<infiniopBatchNormBackwardDescriptor_t>(desc);
            return INFINI_STATUS_SUCCESS;
        }
#endif
    }

#undef CREATE

    return INFINI_STATUS_NOT_IMPLEMENTED;
}

__C infiniStatus_t infiniopGetBatchNormBackwardWorkspaceSize(
    infiniopBatchNormBackwardDescriptor_t desc,
    size_t *size) {
    
    if (!desc || !size) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    auto descriptor = reinterpret_cast<InfiniopDescriptor*>(desc);
    
#define GET_WORKSPACE_SIZE(CASE, NAMESPACE) \
    case CASE: \
        return reinterpret_cast<op::batch_norm_backward::NAMESPACE::Descriptor*>(desc)->get_workspace_size(size)

    switch (descriptor->device_type) {
#ifdef ENABLE_CPU_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ASCEND_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_MOORE, musa);
#endif
#ifdef ENABLE_KUNLUN_API
        GET_WORKSPACE_SIZE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
    }

#undef GET_WORKSPACE_SIZE

    return INFINI_STATUS_NOT_IMPLEMENTED;
}

__C infiniStatus_t infiniopBatchNormBackward(
    infiniopBatchNormBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *input_grad,
    void *weight_grad,
    void *bias_grad,
    const void *output_grad,
    const void *input,
    const void *weight,
    const void *running_mean,
    const void *running_var,
    void *stream) {
    
    if (!desc || !input_grad || !output_grad || !input || !weight || 
        !running_mean || !running_var) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    auto descriptor = reinterpret_cast<InfiniopDescriptor*>(desc);
    
#define CALCULATE(CASE, NAMESPACE) \
    case CASE: \
        return reinterpret_cast<op::batch_norm_backward::NAMESPACE::Descriptor*>(desc)->calculate( \
            workspace, workspace_size, input_grad, weight_grad, bias_grad, \
            output_grad, input, weight, running_mean, running_var, stream)

    switch (descriptor->device_type) {
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
        CALCULATE(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, musa);
#endif
#ifdef ENABLE_KUNLUN_API
        CALCULATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
    }

#undef CALCULATE

    return INFINI_STATUS_NOT_IMPLEMENTED;
}

__C infiniStatus_t infiniopDestroyBatchNormBackwardDescriptor(
    infiniopBatchNormBackwardDescriptor_t desc) {
    
    if (!desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    auto descriptor = reinterpret_cast<InfiniopDescriptor*>(desc);
    
#define DESTROY(CASE, NAMESPACE) \
    case CASE: \
        delete reinterpret_cast<op::batch_norm_backward::NAMESPACE::Descriptor*>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (descriptor->device_type) {
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
        DESTROY(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        DESTROY(INFINI_DEVICE_MOORE, musa);
#endif
#ifdef ENABLE_KUNLUN_API
        DESTROY(INFINI_DEVICE_KUNLUN, kunlun);
#endif
    }

#undef DESTROY

    return INFINI_STATUS_NOT_IMPLEMENTED;
}