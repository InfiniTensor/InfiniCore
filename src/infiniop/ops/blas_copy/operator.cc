#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/blas_copy.h"

#ifdef ENABLE_CPU_API
#include "cpu/blas_copy_cpu.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/blas_copy_metax.h"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/blas_copy_bang.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateBlasCopyDescriptor(
    infiniopHandle_t handle,
    infiniopBlasCopyDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {

#define CREATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                   \
        return op::blas_copy::NAMESPACE::Descriptor::create(                     \
            handle,                                                              \
            reinterpret_cast<op::blas_copy::NAMESPACE::Descriptor **>(desc_ptr), \
            x_desc, y_desc)

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

__INFINI_C infiniStatus_t infiniopGetBlasCopyWorkspaceSize(infiniopBlasCopyDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                     \
    case CASE:                                                                                   \
        *size = reinterpret_cast<op::blas_copy::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopBlasCopy(
    infiniopBlasCopyDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    const void *x,
    void *y,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                      \
        return reinterpret_cast<const op::blas_copy::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, x, y, stream)

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

__INFINI_C infiniStatus_t infiniopDestroyBlasCopyDescriptor(infiniopBlasCopyDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        delete reinterpret_cast<const op::blas_copy::NAMESPACE::Descriptor *>(desc); \
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
