#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/batch_norm.h"

#ifdef ENABLE_CPU_API
#include "cpu/batch_norm_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/batch_norm_nvidia.cuh"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/batch_norm_ascend.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/batch_norm_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "musa/batch_norm_musa.cuh"
#endif
#ifdef ENABLE_KUNLUN_API
#include "kunlun/batch_norm_kunlun.h"
#endif

__C infiniStatus_t infiniopCreateBatchNormDescriptor(
    infiniopHandle_t handle,
    infiniopBatchNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    infiniopTensorDescriptor_t running_mean_desc,
    infiniopTensorDescriptor_t running_var_desc,
    float momentum,
    float eps) {
    
    if (!handle || !desc_ptr || !output_desc || !input_desc || 
        !weight_desc || !bias_desc || !running_mean_desc || !running_var_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }

#define CREATE(CASE, NAMESPACE) \
    case CASE: \
        return op::batch_norm::NAMESPACE::Descriptor::create( \
            handle, reinterpret_cast<op::batch_norm::NAMESPACE::Descriptor**>(desc_ptr), \
            output_desc, input_desc, weight_desc, bias_desc, \
            running_mean_desc, running_var_desc, momentum, eps)

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
#ifdef ENABLE_KUNLUN_API
        CREATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_ASCEND_API
        CREATE(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, musa);
#endif
    }

#undef CREATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetBatchNormWorkspaceSize(infiniopBatchNormDescriptor_t desc, size_t *size) {
    if (!desc || !size) {
        return INFINI_STATUS_BAD_PARAM;
    }

#define GET(CASE, NAMESPACE) \
    case CASE: \
        return reinterpret_cast<op::batch_norm::NAMESPACE::Descriptor*>(desc)->get_workspace_size(size)

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

__C infiniStatus_t infiniopBatchNorm(infiniopBatchNormDescriptor_t desc,
                                     void *workspace,
                                     size_t workspace_size,
                                     void *output,
                                     const void *input,
                                     const void *weight,
                                     const void *bias,
                                     void *running_mean,
                                     void *running_var,
                                     void *stream) {
    if (!desc || !output || !input || !weight || !bias || !running_mean || !running_var) {
        return INFINI_STATUS_BAD_PARAM;
    }

#define CALCULATE(CASE, NAMESPACE) \
    case CASE: \
        return reinterpret_cast<op::batch_norm::NAMESPACE::Descriptor*>(desc)->calculate( \
            workspace, workspace_size, output, input, weight, bias, running_mean, running_var, stream)

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
#ifdef ENABLE_KUNLUN_API
        CALCULATE(INFINI_DEVICE_KUNLUN, kunlun);
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
    }

#undef CALCULATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyBatchNormDescriptor(infiniopBatchNormDescriptor_t desc) {
    if (!desc) {
        return INFINI_STATUS_BAD_PARAM;
    }

#define DESTROY(CASE, NAMESPACE) \
    case CASE: \
        delete reinterpret_cast<op::batch_norm::NAMESPACE::Descriptor *>(desc); \
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
#ifdef ENABLE_KUNLUN_API
        DESTROY(INFINI_DEVICE_KUNLUN, kunlun);
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
    }

#undef DESTROY

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}