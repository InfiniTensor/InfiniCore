#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/linear.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API)
#include "nvidia/linear_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateLinearDescriptor(infiniopHandle_t handle,
                                                  infiniopLinearDescriptor_t *desc_ptr,
                                                  infiniopTensorDescriptor_t d_desc,
                                                  infiniopTensorDescriptor_t c_desc,
                                                  infiniopTensorDescriptor_t bias_desc,
                                                  infiniopTensorDescriptor_t x_desc,
                                                  infiniopTensorDescriptor_t x_scale_desc,
                                                  infiniopTensorDescriptor_t x_zero_desc,
                                                  infiniopTensorDescriptor_t weights_desc,
                                                  infiniopTensorDescriptor_t weights_scale_desc,
                                                  infiniopTensorDescriptor_t weights_zero_desc,
                                                  float alpha,
                                                  float beta) {
#define CREATE(CASE, NAMESPACE)                                               \
    case CASE:                                                                \
        return op::linear::NAMESPACE::Descriptor::create(                     \
            handle,                                                           \
            reinterpret_cast<op::linear::NAMESPACE::Descriptor **>(desc_ptr), \
            d_desc,                                                           \
            c_desc,                                                           \
            bias_desc,                                                        \
            x_desc,                                                           \
            x_scale_desc,                                                     \
            x_zero_desc,                                                      \
            weights_desc,                                                     \
            weights_scale_desc,                                               \
            weights_zero_desc,                                                \
            alpha,                                                            \
            beta);
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__C infiniStatus_t infiniopGetLinearWorkspaceSize(infiniopLinearDescriptor_t desc, size_t *size) {
    switch (desc->device_type) {
#define GET(CASE, NAMESPACE)                                                                     \
    case CASE:                                                                                   \
        *size = reinterpret_cast<op::linear::NAMESPACE::Descriptor *>(desc)->minWorkspaceSize(); \
        return INFINI_STATUS_SUCCESS;
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__C infiniStatus_t infiniopLinear(infiniopLinearDescriptor_t desc,
                                  void *workspace,
                                  size_t workspace_size,
                                  void *d,
                                  const void *c,
                                  const void *bias,
                                  const void *x,
                                  const void *x_scale,
                                  const void *x_zero,
                                  const void *weights,
                                  const void *weights_scale,
                                  const void *weights_zero,
                                  void *stream) {
#define CACULATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                         \
        return reinterpret_cast<op::linear::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, d, c, bias, x, x_scale, x_zero, weights, weights_scale, weights_zero, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CACULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        CACULATE(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CACULATE
}

__C infiniStatus_t infiniopDestroyLinearDescriptor(infiniopLinearDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                            \
    case CASE:                                                              \
        delete reinterpret_cast<op::linear::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
