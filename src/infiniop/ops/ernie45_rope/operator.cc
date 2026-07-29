#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/ernie45_rope.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API) || defined(ENABLE_ALI_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/ernie45_rope_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateErnie45MropeDescriptor(
    infiniopHandle_t handle,
    infiniopErnie45MropeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t q,
    infiniopTensorDescriptor_t k,
    infiniopTensorDescriptor_t positions,
    double rope_theta,
    size_t section_h,
    size_t section_w,
    size_t section_t) {
#define CREATE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                           \
        return op::ernie45_rope::NAMESPACE::MropeDescriptor::create(                     \
            handle,                                                                      \
            reinterpret_cast<op::ernie45_rope::NAMESPACE::MropeDescriptor **>(desc_ptr), \
            q, k, positions, rope_theta, section_h, section_w, section_t)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetErnie45MropeWorkspaceSize(infiniopErnie45MropeDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                   \
    case CASE:                                                                                                 \
        *size = reinterpret_cast<const op::ernie45_rope::NAMESPACE::MropeDescriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopErnie45Mrope(
    infiniopErnie45MropeDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *q,
    void *k,
    const void *positions,
    void *stream) {
#define CALC(CASE, NAMESPACE)                                                                           \
    case CASE:                                                                                          \
        return reinterpret_cast<const op::ernie45_rope::NAMESPACE::MropeDescriptor *>(desc)->calculate( \
            workspace, workspace_size, q, k, positions, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALC(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALC(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALC(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALC(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALC
}

__INFINI_C infiniStatus_t infiniopDestroyErnie45MropeDescriptor(infiniopErnie45MropeDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                             \
    case CASE:                                                                               \
        delete reinterpret_cast<const op::ernie45_rope::NAMESPACE::MropeDescriptor *>(desc); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}

__INFINI_C infiniStatus_t infiniopCreateErnie45VisionRopeDescriptor(
    infiniopHandle_t handle,
    infiniopErnie45VisionRopeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t q,
    infiniopTensorDescriptor_t k,
    infiniopTensorDescriptor_t positions,
    double rope_theta) {
#define CREATE(CASE, NAMESPACE)                                                               \
    case CASE:                                                                                \
        return op::ernie45_rope::NAMESPACE::VisionRopeDescriptor::create(                     \
            handle,                                                                           \
            reinterpret_cast<op::ernie45_rope::NAMESPACE::VisionRopeDescriptor **>(desc_ptr), \
            q, k, positions, rope_theta)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetErnie45VisionRopeWorkspaceSize(infiniopErnie45VisionRopeDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                        \
    case CASE:                                                                                                      \
        *size = reinterpret_cast<const op::ernie45_rope::NAMESPACE::VisionRopeDescriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopErnie45VisionRope(
    infiniopErnie45VisionRopeDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *q,
    void *k,
    const void *positions,
    void *stream) {
#define CALC(CASE, NAMESPACE)                                                                                \
    case CASE:                                                                                               \
        return reinterpret_cast<const op::ernie45_rope::NAMESPACE::VisionRopeDescriptor *>(desc)->calculate( \
            workspace, workspace_size, q, k, positions, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALC(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALC(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALC(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALC(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALC
}

__INFINI_C infiniStatus_t infiniopDestroyErnie45VisionRopeDescriptor(infiniopErnie45VisionRopeDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                                  \
    case CASE:                                                                                    \
        delete reinterpret_cast<const op::ernie45_rope::NAMESPACE::VisionRopeDescriptor *>(desc); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
