#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/rotmg.h"

#ifdef ENABLE_CPU_API
#include "cpu/rotmg_cpu.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/rotmg_metax.h"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/rotmg_bang.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateRotmgDescriptor(
    infiniopHandle_t handle,
    infiniopRotmgDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t d1_desc,
    infiniopTensorDescriptor_t d2_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t y1_desc,
    infiniopTensorDescriptor_t param_desc) {

#define CREATE(CASE, NAMESPACE)                                              \
    case CASE:                                                               \
        return op::rotmg::NAMESPACE::Descriptor::create(                     \
            handle,                                                          \
            reinterpret_cast<op::rotmg::NAMESPACE::Descriptor **>(desc_ptr), \
            d1_desc, d2_desc, x1_desc, y1_desc, param_desc)

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

__INFINI_C infiniStatus_t infiniopGetRotmgWorkspaceSize(infiniopRotmgDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                 \
    case CASE:                                                                               \
        *size = reinterpret_cast<op::rotmg::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopRotmg(
    infiniopRotmgDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *d1,
    void *d2,
    void *x1,
    const void *y1,
    void *param,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                              \
    case CASE:                                                                  \
        return reinterpret_cast<const op::rotmg::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, d1, d2, x1, y1, param, stream)

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

__INFINI_C infiniStatus_t infiniopDestroyRotmgDescriptor(infiniopRotmgDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                   \
        delete reinterpret_cast<const op::rotmg::NAMESPACE::Descriptor *>(desc); \
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
