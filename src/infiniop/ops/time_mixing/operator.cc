#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/time_mixing.h"

#ifdef ENABLE_ASCEND_API
#include "ascend/time_mixing_ascend.h"
#endif

__C infiniStatus_t infiniopCreateTimeMixingDescriptor(
    infiniopHandle_t handle,
    infiniopTimeMixingDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t r_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

#define CREATE(CASE, NAMESPACE)                                                    \
    case CASE:                                                                     \
        return op::time_mixing::NAMESPACE::Descriptor::create(                     \
            handle,                                                                \
            reinterpret_cast<op::time_mixing::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                                \
            r_desc,                                                                \
            w_desc,                                                                \
            k_desc,                                                                \
            v_desc,                                                                \
            a_desc,                                                                \
            b_desc)

    switch (handle->device) {

#ifdef ENABLE_ASCEND_API
        CREATE(INFINI_DEVICE_ASCEND, ascend);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetTimeMixingWorkspaceSize(infiniopTimeMixingDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                       \
    case CASE:                                                                                     \
        *size = reinterpret_cast<op::time_mixing::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_ASCEND_API
        GET(INFINI_DEVICE_ASCEND, ascend)
#endif
    }

#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopTimeMixing(
    infiniopTimeMixingDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    void const *r,
    void const *w,
    void const *k,
    void const *v,
    void const *a,
    void const *b,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                    \
    case CASE:                                                                        \
        return reinterpret_cast<const op::time_mixing::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, y, r, w, k, v, a, b, stream)

    switch (desc->device_type) {

#ifdef ENABLE_ASCEND_API
        CALCULATE(INFINI_DEVICE_ASCEND, ascend);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyTimeMixingDescriptor(infiniopTimeMixingDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                        \
    case CASE:                                                                         \
        delete reinterpret_cast<const op::time_mixing::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_ASCEND_API
        DELETE(INFINI_DEVICE_ASCEND, ascend)
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}