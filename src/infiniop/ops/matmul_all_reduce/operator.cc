#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/matmul_all_reduce.h"

#ifdef ENABLE_ASCEND_API
#include "ascend/matmul_all_reduce_ascend.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateMatmulAllReduceDescriptor(
    infiniopHandle_t handle,
    infiniopMatmulAllReduceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    const char *group_name) {
    switch (handle->device) {
#ifdef ENABLE_ASCEND_API
    case INFINI_DEVICE_ASCEND:
        return op::matmul_all_reduce::ascend::Descriptor::create(
            handle,
            reinterpret_cast<op::matmul_all_reduce::ascend::Descriptor **>(
                desc_ptr),
            output_desc, input_desc, weight_desc, bias_desc, group_name);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__INFINI_C infiniStatus_t infiniopGetMatmulAllReduceWorkspaceSize(
    infiniopMatmulAllReduceDescriptor_t desc,
    size_t *size) {
    if (desc == nullptr || size == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    switch (desc->device_type) {
#ifdef ENABLE_ASCEND_API
    case INFINI_DEVICE_ASCEND:
        *size = reinterpret_cast<
                    const op::matmul_all_reduce::ascend::Descriptor *>(desc)
                    ->workspaceSize();
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__INFINI_C infiniStatus_t infiniopMatmulAllReduce(
    infiniopMatmulAllReduceDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *weight,
    const void *bias,
    void *stream) {
    if (desc == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    switch (desc->device_type) {
#ifdef ENABLE_ASCEND_API
    case INFINI_DEVICE_ASCEND:
        return reinterpret_cast<
                   const op::matmul_all_reduce::ascend::Descriptor *>(desc)
            ->calculate(
                workspace, workspace_size, output, input, weight, bias,
                stream);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__INFINI_C infiniStatus_t infiniopDestroyMatmulAllReduceDescriptor(
    infiniopMatmulAllReduceDescriptor_t desc) {
    if (desc == nullptr) {
        return INFINI_STATUS_SUCCESS;
    }
    switch (desc->device_type) {
#ifdef ENABLE_ASCEND_API
    case INFINI_DEVICE_ASCEND:
        delete reinterpret_cast<
            const op::matmul_all_reduce::ascend::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
