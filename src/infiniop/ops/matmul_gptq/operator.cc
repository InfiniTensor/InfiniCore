#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/matmul_gptq.h"

#ifdef ENABLE_CPU_API
#include "cpu/matmul_gptq_cpu.h"
#endif

__C infiniStatus_t infiniopCreateMatmulGptqDescriptor(infiniopHandle_t handle,
                                                      infiniopMatmulGptqDescriptor_t *desc_ptr,
                                                      infiniopTensorDescriptor_t c_desc,
                                                      infiniopTensorDescriptor_t a_desc,
                                                      infiniopTensorDescriptor_t packed_weights_desc,
                                                      infiniopTensorDescriptor_t b_scale_desc,
                                                      infiniopTensorDescriptor_t zero_desc) {
    switch (handle->device) {
#ifdef ENABLE_CPU_API
    case INFINI_DEVICE_CPU:
        return op::matmul_gptq::cpu::Descriptor::create(handle, reinterpret_cast<op::matmul_gptq::cpu::Descriptor **>(desc_ptr), c_desc, a_desc, packed_weights_desc, b_scale_desc, zero_desc);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopGetMatmulGptqWorkspaceSize(infiniopMatmulGptqDescriptor_t desc, size_t *size) {
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
    case INFINI_DEVICE_CPU:
        *size = reinterpret_cast<op::matmul_gptq::cpu::Descriptor *>(desc)->minWorkspaceSize();
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopMatmulQuant(infiniopMatmulGptqDescriptor_t desc,
                                       void *workspace,
                                       size_t workspace_size,
                                       void *packed_weights,
                                       void *b_scale,
                                       void *zero,
                                       const void *a,
                                       const void *b,
                                       void *stream) {
    switch (desc->device_type) {
    case INFINI_DEVICE_CPU:
#ifdef ENABLE_CPU_API
        return reinterpret_cast<op::matmul_gptq::cpu::Descriptor *>(desc)->quant(workspace, workspace_size, packed_weights, b_scale, zero, a, b, stream);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopMatmulGptq(infiniopMatmulGptqDescriptor_t desc,
                                      void *workspace,
                                      size_t workspace_size,
                                      void *c,
                                      const void *a,
                                      void *packed_weights,
                                      void *b_scale,
                                      void *zero,
                                      void *stream) {
    switch (desc->device_type) {
    case INFINI_DEVICE_CPU:
#ifdef ENABLE_CPU_API
        return reinterpret_cast<op::matmul_gptq::cpu::Descriptor *>(desc)->calculate(workspace, workspace_size, c, a, packed_weights, b_scale, zero, stream);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopDestroyMatmulGptqDescriptor(infiniopMatmulGptqDescriptor_t desc) {
    switch (desc->device_type) {
    case INFINI_DEVICE_CPU:
#ifdef ENABLE_CPU_API
        delete reinterpret_cast<op::matmul_gptq::cpu::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
