#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/fused_moe_mxfp4.h"

#ifdef ENABLE_CPU_API
#include "cpu/fused_moe_mxfp4_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
#include "nvidia/fused_moe_mxfp4_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/fused_moe_mxfp4_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/fused_moe_mxfp4_moore.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateFusedMoeMxfp4Descriptor(
    infiniopHandle_t handle,
    infiniopFusedMoeMxfp4Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t selected_experts_desc,
    infiniopTensorDescriptor_t routing_weights_desc,
    infiniopTensorDescriptor_t w13_packed_desc,
    infiniopTensorDescriptor_t w13_scale_desc,
    infiniopTensorDescriptor_t w2_packed_desc,
    infiniopTensorDescriptor_t w2_scale_desc,
    infiniopFusedMoeActivation_t activation) {
#define CREATE(CASE, NAMESPACE)                                                        \
    case CASE:                                                                         \
        return op::fused_moe_mxfp4::NAMESPACE::Descriptor::create(                     \
            handle,                                                                    \
            reinterpret_cast<op::fused_moe_mxfp4::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc, input_desc, selected_experts_desc, routing_weights_desc,      \
            w13_packed_desc, w13_scale_desc, w2_packed_desc, w2_scale_desc, activation)
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

__INFINI_C infiniStatus_t infiniopGetFusedMoeMxfp4WorkspaceSize(
    infiniopFusedMoeMxfp4Descriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                               \
    case CASE:                                                                             \
        *size = reinterpret_cast<const op::fused_moe_mxfp4::NAMESPACE::Descriptor *>(desc) \
                    ->workspaceSize();                                                     \
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

__INFINI_C infiniStatus_t infiniopFusedMoeMxfp4(
    infiniopFusedMoeMxfp4Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *selected_experts,
    const void *routing_weights,
    const void *w13_packed,
    const void *w13_scale,
    const void *w2_packed,
    const void *w2_scale,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                        \
    case CASE:                                                                            \
        return reinterpret_cast<const op::fused_moe_mxfp4::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, output, input, selected_experts,       \
                        routing_weights, w13_packed, w13_scale, w2_packed, w2_scale, stream)
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

__INFINI_C infiniStatus_t infiniopDestroyFusedMoeMxfp4Descriptor(
    infiniopFusedMoeMxfp4Descriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                           \
    case CASE:                                                                             \
        delete reinterpret_cast<const op::fused_moe_mxfp4::NAMESPACE::Descriptor *>(desc); \
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
