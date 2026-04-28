#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/scal.h"

#ifdef ENABLE_CPU_API
#include "cpu/scal_cpu.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/scal_metax.h"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/scal_bang.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateScalDescriptor(
    infiniopHandle_t handle,
    infiniopScalDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc) {

#define CREATE(CASE, NAMESPACE)                                             \
    case CASE:                                                              \
        return op::scal::NAMESPACE::Descriptor::create(                     \
            handle,                                                         \
            reinterpret_cast<op::scal::NAMESPACE::Descriptor **>(desc_ptr), \
            alpha_desc, x_desc)

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

__INFINI_C infiniStatus_t infiniopGetScalWorkspaceSize(infiniopScalDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                \
    case CASE:                                                                              \
        *size = reinterpret_cast<op::scal::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopScal(
    infiniopScalDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    void *x,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::scal::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, alpha, x, stream)

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

__INFINI_C infiniStatus_t infiniopDestroyScalDescriptor(infiniopScalDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        delete reinterpret_cast<const op::scal::NAMESPACE::Descriptor *>(desc); \
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
