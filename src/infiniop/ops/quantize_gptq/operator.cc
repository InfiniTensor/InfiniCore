#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/quantize_gptq.h"

#ifdef ENABLE_CPU_API
#include "cpu/quantize_gptq_cpu.h"
#endif
#ifdef ENABLE_CUDA_API
#include "cuda/quantize_gptq_cuda.cuh"
#endif

__C infiniStatus_t infiniopCreateQuantizeGPTQDescriptor(infiniopHandle_t handle,
                                                        infiniopQuantizeGPTQDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t c_desc,
                                                        infiniopTensorDescriptor_t a_desc,
                                                        infiniopTensorDescriptor_t packed_weights_desc,
                                                        infiniopTensorDescriptor_t b_scale_desc,
                                                        infiniopTensorDescriptor_t zero_desc) {
#define CREATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        return op::quantize_gptq::NAMESPACE::Descriptor::create(                     \
            handle,                                                                  \
            reinterpret_cast<op::quantize_gptq::NAMESPACE::Descriptor **>(desc_ptr), \
            c_desc,                                                                  \
            a_desc,                                                                  \
            packed_weights_desc,                                                     \
            b_scale_desc,                                                            \
            zero_desc);
    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_CUDA_API
        CREATE(INFINI_DEVICE_NVIDIA, cuda)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopGetQuantizeGPTQWorkspaceSize(infiniopQuantizeGPTQDescriptor_t desc, size_t *size) {
    switch (desc->device_type) {
#define GET(CASE, NAMESPACE)                                                                            \
    case CASE:                                                                                          \
        *size = reinterpret_cast<op::quantize_gptq::NAMESPACE::Descriptor *>(desc)->minWorkspaceSize(); \
        return INFINI_STATUS_SUCCESS;
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_CUDA_API
        GET(INFINI_DEVICE_NVIDIA, cuda)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopQuantizeGPTQ(infiniopQuantizeGPTQDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *packed_weights,
                                        void *b_scale,
                                        void *zero,
                                        const void *a,
                                        const void *b,
                                        void *stream) {
#define QUANT(CASE, NAMESPACE)                                                            \
    case CASE:                                                                            \
        return reinterpret_cast<op::quantize_gptq::NAMESPACE::Descriptor *>(desc)->quant( \
            workspace, workspace_size, packed_weights, b_scale, zero, a, b, stream);

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        QUANT(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_CUDA_API
        QUANT(INFINI_DEVICE_NVIDIA, cuda)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopQuantizeLinearGPTQ(infiniopQuantizeGPTQDescriptor_t desc,
                                              void *workspace,
                                              size_t workspace_size,
                                              void *c,
                                              const void *a,
                                              void *packed_weights,
                                              void *b_scale,
                                              void *zero,
                                              void *stream) {
#define CACULATE(CASE, NAMESPACE)                                                             \
    case CASE:                                                                                \
        return reinterpret_cast<op::quantize_gptq::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, c, a, packed_weights, b_scale, zero, stream);

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CACULATE(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_CUDA_API
        CACULATE(INFINI_DEVICE_NVIDIA, cuda)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopDestroyQuantizeGPTQDescriptor(infiniopQuantizeGPTQDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                   \
    case CASE:                                                                     \
        delete reinterpret_cast<op::quantize_gptq::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_CUDA_API
        DESTROY(INFINI_DEVICE_NVIDIA, cuda)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
