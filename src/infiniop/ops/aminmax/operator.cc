#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/aminmax.h"

#ifdef ENABLE_CPU_API
#include "cpu/aminmax_cpu.h"
#endif

__C infiniStatus_t infiniopCreateAminmaxDescriptor(
    infiniopHandle_t handle,
    infiniopAminmaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t min_output_desc,
    infiniopTensorDescriptor_t max_output_desc,
    infiniopTensorDescriptor_t input_desc,
    int64_t dim,
    int32_t keepdim,
    int32_t has_dim) {

#define CREATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        return op::aminmax::NAMESPACE::Descriptor::create(                     \
            handle,                                                            \
            reinterpret_cast<op::aminmax::NAMESPACE::Descriptor **>(desc_ptr), \
            min_output_desc,                                                   \
            max_output_desc,                                                   \
            input_desc,                                                        \
            dim, keepdim, has_dim)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetAminmaxWorkspaceSize(infiniopAminmaxDescriptor_t desc,
                                                   size_t *size) {
    *size = 0;
    return INFINI_STATUS_SUCCESS;
}

__C infiniStatus_t infiniopAminmax(infiniopAminmaxDescriptor_t desc,
                                   void *workspace,
                                   size_t workspace_size,
                                   void *min_output,
                                   void *max_output,
                                   const void *input,
                                   void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                    \
        return reinterpret_cast<const op::aminmax::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, min_output, max_output, input, stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyAminmaxDescriptor(infiniopAminmaxDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                    \
    case CASE:                                                                     \
        delete reinterpret_cast<const op::aminmax::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
