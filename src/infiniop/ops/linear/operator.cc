#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/linear.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateLinearDescriptor(
    infiniopHandle_t handle, infiniopLinearDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t d_desc, infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc, infiniopTensorDescriptor_t c_desc) {

#define CREATE(CASE, NAMESPACE)                                               \
    case CASE:                                                                \
        return op::linear::NAMESPACE::Descriptor::create(                     \
            handle,                                                           \
            reinterpret_cast<op::linear::NAMESPACE::Descriptor **>(desc_ptr), \
            d_desc, a_desc, b_desc, c_desc)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t
infiniopGetLinearWorkspaceSize(infiniopLinearDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                      \
    case CASE:                                                                    \
        *size = reinterpret_cast<const op::linear::NAMESPACE::Descriptor *>(desc) \
                    ->workspaceSize();                                            \
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

__C infiniStatus_t infiniopLinear(
    infiniopLinearDescriptor_t desc, float alpha, const void *a,
    const void *a_scale, const void *b, const void *b_scale, float beta,
    const void *c, const void *c_scale, const void *bias, void *d,
    const void *d_scale, bool is_blockwise, bool is_a_1d_scaled,
    bool is_b_1d_scaled, void *workspace, size_t workspace_size, void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                    \
        return reinterpret_cast<const op::linear::NAMESPACE::Descriptor *>(desc)  \
            ->calculate(alpha, a, a_scale, b, b_scale, beta, c, c_scale, bias, d, \
                        d_scale, is_blockwise, is_a_1d_scaled, is_b_1d_scaled,    \
                        workspace, workspace_size, stream)
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
infiniopDestroyLinearDescriptor(infiniopLinearDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                    \
        delete reinterpret_cast<const op::linear::NAMESPACE::Descriptor *>(desc); \
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