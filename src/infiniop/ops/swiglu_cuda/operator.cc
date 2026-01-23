#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/swiglu_cuda.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API)
#include "nvidia/swiglu_cuda_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/swiglu_cuda_metax.cuh"
#endif

__C infiniStatus_t infiniopCreateSwiGLUCudaDescriptor(
    infiniopHandle_t handle,
    infiniopSwiGLUCudaDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

#define CREATE(CASE, NAMESPACE)                                                    \
    case CASE:                                                                     \
        return op::swiglu_cuda::NAMESPACE::Descriptor::create(                     \
            handle,                                                                \
            reinterpret_cast<op::swiglu_cuda::NAMESPACE::Descriptor **>(desc_ptr), \
            c_desc,                                                                \
            a_desc,                                                                \
            b_desc)

    switch (handle->device) {

#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetSwiGLUCudaWorkspaceSize(infiniopSwiGLUCudaDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                       \
    case CASE:                                                                                     \
        *size = reinterpret_cast<op::swiglu_cuda::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__C infiniStatus_t infiniopSwiGLUCuda(
    infiniopSwiGLUCudaDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                    \
    case CASE:                                                                        \
        return reinterpret_cast<const op::swiglu_cuda::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, c, a, b, stream)

    switch (desc->device_type) {

#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroySwiGLUCudaDescriptor(infiniopSwiGLUCudaDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                        \
    case CASE:                                                                         \
        delete reinterpret_cast<const op::swiglu_cuda::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
