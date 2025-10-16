#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/quantize.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/quantize_group_8bit_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateQuantizeDescriptor(
    infiniopHandle_t handle, infiniopQuantizeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t output_q_desc,
    infiniopTensorDescriptor_t output_s_desc) {

#define CREATE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        return op::quantize::NAMESPACE::Descriptor::create(                     \
            handle,                                                             \
            reinterpret_cast<op::quantize::NAMESPACE::Descriptor **>(desc_ptr), \
            input_desc, output_q_desc, output_s_desc)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetQuantizeWorkspaceSize(
    infiniopQuantizeDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                        \
    case CASE:                                                                      \
        *size = reinterpret_cast<const op::quantize::NAMESPACE::Descriptor *>(desc) \
                    ->workspaceSize();                                              \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__C infiniStatus_t infiniopQuantize(infiniopQuantizeDescriptor_t desc,
                                    void *workspace, size_t workspace_size,
                                    void *input, void *output_q, void *output_s,
                                    int group_size, double eps, double min_8bit,
                                    double max_8bit, bool scale_ue8m0,
                                    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                     \
        return reinterpret_cast<const op::quantize::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, input, output_q, output_s,      \
                        group_size, eps, min_8bit, max_8bit, scale_ue8m0, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyQuantizeDescriptor(infiniopQuantizeDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                               \
    case CASE:                                                                \
        delete reinterpret_cast<const op::quantize::NAMESPACE::Descriptor *>( \
            desc);                                                            \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}

// #endif