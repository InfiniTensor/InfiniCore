#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/random_sample_batched.h"

#ifdef ENABLE_CPU_API
// #include "cpu/random_sample_batched_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
// #include "nvidia/random_sample_batched_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateRandomSampleBatchedDescriptor(
    infiniopHandle_t handle,
    infiniopRandomSampleBatchedDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t result,
    infiniopTensorDescriptor_t probs) {

#define CREATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        return op::random_sample::NAMESPACE::Descriptor::create(                     \
            handle,                                                                  \
            reinterpret_cast<op::random_sample::NAMESPACE::Descriptor **>(desc_ptr), \
            result,                                                                  \
            probs)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        // CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
        // CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetRandomSampleBatchedWorkspaceSize(
    infiniopRandomSampleBatchedDescriptor_t desc,
    size_t *size) {

#define GET_SIZE(CASE, NAMESPACE)                                     \
    case CASE:                                                        \
        using Ptr = const op::random_sample::NAMESPACE::Descriptor *; \
        *size = reinterpret_cast<Ptr>(desc)->minWorkspaceSize();      \
        }                                                             \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        // GET_SIZE(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
        // GET_SIZE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET_SIZE
}

__C infiniStatus_t infiniopRandomSampleBatched(
    infiniopRandomSampleBatchedDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    const float *random_val,
    const float *topp,
    const int *topk,
    const float *temperature,
    int batch_size,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                          \
        return reinterpret_cast<const op::random_sample::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size,                                      \
                        result, probs,                                                  \
                        random_val,                                                     \
                        topp, topk, temperature,                                        \
                        batch_size,                                                     \
                        stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        // CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
        // CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyRandomSampleBatchedDescriptor(
    infiniopRandomSampleBatchedDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                           \
        delete reinterpret_cast<const op::random_sample::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        // DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
        // DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
