#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/blas_amax.h"

#ifdef ENABLE_CPU_API
#include "cpu/blas_amax_cpu.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/blas_amax_metax.h"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/blas_amax_bang.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateBlasAmaxDescriptor(
    infiniopHandle_t handle,
    infiniopBlasAmaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t result_desc) {

#define CREATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                   \
        return op::blas_amax::NAMESPACE::Descriptor::create(                     \
            handle,                                                              \
            reinterpret_cast<op::blas_amax::NAMESPACE::Descriptor **>(desc_ptr), \
            x_desc,                                                              \
            result_desc)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_CAMBRICON_API
        CREATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetBlasAmaxWorkspaceSize(infiniopBlasAmaxDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                     \
    case CASE:                                                                                   \
        *size = reinterpret_cast<op::blas_amax::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_CAMBRICON_API
        GET(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__INFINI_C infiniStatus_t infiniopBlasAmax(
    infiniopBlasAmaxDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    const void *x,
    void *result,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                      \
        return reinterpret_cast<const op::blas_amax::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, x, result, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_CAMBRICON_API
        CALCULATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyBlasAmaxDescriptor(infiniopBlasAmaxDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        delete reinterpret_cast<const op::blas_amax::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_CAMBRICON_API
        DELETE(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
