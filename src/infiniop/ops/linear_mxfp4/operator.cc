#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/linear_mxfp4.h"

#ifdef ENABLE_CPU_API
#include "cpu/linear_mxfp4_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
#include "nvidia/linear_mxfp4_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/linear_mxfp4_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/linear_mxfp4_moore.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateLinearMxfp4Descriptor(
    infiniopHandle_t handle,
    infiniopLinearMxfp4Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t packed_weight_desc,
    infiniopTensorDescriptor_t weight_scale_desc,
    infiniopTensorDescriptor_t bias_desc,
    float alpha) {
#define CREATE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                      \
        return op::linear_mxfp4::NAMESPACE::Descriptor::create(                     \
            handle,                                                                 \
            reinterpret_cast<op::linear_mxfp4::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc, input_desc, packed_weight_desc, weight_scale_desc,         \
            bias_desc, alpha)
    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetLinearMxfp4WorkspaceSize(
    infiniopLinearMxfp4Descriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                            \
    case CASE:                                                                          \
        *size = reinterpret_cast<const op::linear_mxfp4::NAMESPACE::Descriptor *>(desc) \
                    ->workspaceSize();                                                  \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopLinearMxfp4(
    infiniopLinearMxfp4Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *packed_weight,
    const void *weight_scale,
    const void *bias,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                         \
        return reinterpret_cast<const op::linear_mxfp4::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, output, input, packed_weight,       \
                        weight_scale, bias, stream)
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyLinearMxfp4Descriptor(
    infiniopLinearMxfp4Descriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                        \
    case CASE:                                                                          \
        delete reinterpret_cast<const op::linear_mxfp4::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        DESTROY(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
