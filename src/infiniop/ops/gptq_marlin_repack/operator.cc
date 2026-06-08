#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/gptq_marlin_repack.h"

#if defined ENABLE_NVIDIA_API
#include "nvidia/gptq_marlin_repack_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateGptqMarlinRepackDescriptor(
    infiniopHandle_t handle,
    infiniopGptqMarlinRepackDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t perm_desc,
    int64_t num_bits,
    bool is_a_8bit) {
#define CREATE(CASE, NAMESPACE)                                                           \
    case CASE:                                                                            \
        return op::gptq_marlin_repack::NAMESPACE::Descriptor::create(                     \
            handle,                                                                       \
            reinterpret_cast<op::gptq_marlin_repack::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc,                                                                  \
            input_desc,                                                                   \
            perm_desc,                                                                    \
            num_bits,                                                                     \
            is_a_8bit)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetGptqMarlinRepackWorkspaceSize(infiniopGptqMarlinRepackDescriptor_t desc,
                                                                   size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                    \
    case CASE:                                                                                                  \
        *size = reinterpret_cast<const op::gptq_marlin_repack::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopGptqMarlinRepack(
    infiniopGptqMarlinRepackDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *perm,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                           \
    case CASE:                                                                               \
        return reinterpret_cast<const op::gptq_marlin_repack::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, output, input, perm, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t
infiniopDestroyGptqMarlinRepackDescriptor(infiniopGptqMarlinRepackDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                               \
    case CASE:                                                                                \
        delete reinterpret_cast<const op::gptq_marlin_repack::NAMESPACE::Descriptor *>(desc); \
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
