#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/where.h"

#ifdef ENABLE_CPU_API
#include "cpu/where_cpu.h"
#include "cpu/where_indices_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/where_indices_nvidia.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/where_indices_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/where_indices_moore.h"
#endif

__C infiniStatus_t infiniopCreateWhereDescriptor(
    infiniopHandle_t handle,
    infiniopWhereDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t cond_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {

#define CREATE(CASE, NAMESPACE)                                              \
    case CASE:                                                               \
        return op::where::NAMESPACE::Descriptor::create(                     \
            handle,                                                          \
            reinterpret_cast<op::where::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc,                                                        \
            {cond_desc, x_desc, y_desc})

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetWhereWorkspaceSize(
    infiniopWhereDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                 \
    case CASE:                                                                               \
        *size = reinterpret_cast<op::where::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopWhere(
    infiniopWhereDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *cond,
    const void *x,
    const void *y,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                              \
    case CASE:                                                                  \
        return reinterpret_cast<const op::where::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, out, {cond, x, y}, stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyWhereDescriptor(infiniopWhereDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                   \
        delete reinterpret_cast<const op::where::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}

// where(cond) -> indices tuple
__C infiniStatus_t infiniopCreateWhereIndicesDescriptor(
    infiniopHandle_t handle,
    infiniopWhereIndicesDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t cond_desc) {

#define CREATE_INDICES(CASE, NAMESPACE)                                             \
    case CASE:                                                                      \
        return op::where::NAMESPACE::IndicesDescriptor::create(                     \
            handle,                                                                 \
            reinterpret_cast<op::where::NAMESPACE::IndicesDescriptor **>(desc_ptr), \
            cond_desc)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE_INDICES(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
        CREATE_INDICES(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE_INDICES(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE_INDICES(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE_INDICES(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CREATE_INDICES(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE_INDICES
}

__C infiniStatus_t infiniopGetWhereIndicesWorkspaceSize(
    infiniopWhereIndicesDescriptor_t desc,
    size_t *size) {

#define GET_INDICES(CASE, NAMESPACE)                                                                \
    case CASE:                                                                                      \
        *size = reinterpret_cast<op::where::NAMESPACE::IndicesDescriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET_INDICES(INFINI_DEVICE_CPU, cpu)
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
        GET_INDICES(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        GET_INDICES(INFINI_DEVICE_ILUVATAR, nvidia)
#endif
#ifdef ENABLE_QY_API
        GET_INDICES(INFINI_DEVICE_QY, nvidia)
#endif
#ifdef ENABLE_METAX_API
        GET_INDICES(INFINI_DEVICE_METAX, metax)
#endif
#ifdef ENABLE_MOORE_API
        GET_INDICES(INFINI_DEVICE_MOORE, moore)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET_INDICES
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopWhereIndices(
    infiniopWhereIndicesDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void **outputs,
    const void *cond,
    void *stream,
    size_t *num_true) {

#define CALCULATE_INDICES(CASE, NAMESPACE)                                             \
    case CASE:                                                                         \
        return reinterpret_cast<const op::where::NAMESPACE::IndicesDescriptor *>(desc) \
            ->calculate(workspace, workspace_size, outputs, cond, stream, num_true)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE_INDICES(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
        CALCULATE_INDICES(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE_INDICES(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALCULATE_INDICES(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE_INDICES(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE_INDICES(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE_INDICES
}

__C infiniStatus_t infiniopDestroyWhereIndicesDescriptor(infiniopWhereIndicesDescriptor_t desc) {

#define DELETE_INDICES(CASE, NAMESPACE)                                                 \
    case CASE:                                                                          \
        if (desc != nullptr) {                                                         \
            delete reinterpret_cast<op::where::NAMESPACE::IndicesDescriptor *>(       \
                const_cast<void *>(reinterpret_cast<const void *>(desc)));             \
        }                                                                               \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DELETE_INDICES(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
        DELETE_INDICES(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DELETE_INDICES(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        DELETE_INDICES(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DELETE_INDICES(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        DELETE_INDICES(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE_INDICES
}
