#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/rotary_embedding.h"

__C infiniStatus_t infiniopCreateRoPEDescriptor(
    infiniopHandle_t handle, infiniopRoPEDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t t, infiniopTensorDescriptor_t pos_ids,
    infiniopTensorDescriptor_t sin_table,
    infiniopTensorDescriptor_t cos_table) {
    switch (handle->device) {
#ifdef ENABLE_CPU_API
    case INFINI_DEVICE_CPU:
        return cpuCreateRoPEDescriptor((infiniopCpuHandle_t)handle,
                                       (infiniopRoPECpuDescriptor_t *)desc_ptr, t,
                                       pos_ids, sin_table, cos_table);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu: {
        return cudaCreateRoPEDescriptor((CudaHandle_t)handle,
                                        (infiniopRoPECpuDescriptor_t *)desc_ptr, t,
                                        pos_ids, sin_table, cos_table);
    }

#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangCreateRoPEDescriptor((BangHandle_t)handle,
                                        (RoPEBangDescriptor_t *)desc_ptr, t,
                                        pos_ids, sin_table, cos_table);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu: {
        return ascendCreateRoPEDescriptor((AscendHandle_t)handle,
                                          (RoPEAscendDescriptor_t *)desc_ptr, t,
                                          pos_ids, sin_table, cos_table);
    }
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaCreateRoPEDescriptor((MacaHandle_t)handle,
                                        (RoPEMacaDescriptor_t *)desc_ptr, t,
                                        pos_ids, sin_table, cos_table);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu: {
        return musaCreateRoPEDescriptor((MusaHandle_t)handle,
                                        (RoPEMusaDescriptor_t *)desc_ptr, t,
                                        pos_ids, sin_table, cos_table);
    }
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetRoPEWorkspaceSize(infiniopRoPEDescriptor_t desc,
                                                size_t *size) {
    switch (desc->device_type) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuGetRoPEWorkspaceSize((RoPECpuDescriptor_t)desc, size);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu: {
        return cudaGetRoPEWorkspaceSize((RoPECudaDescriptor_t)desc, size);
    }

#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangGetRoPEWorkspaceSize((RoPEBangDescriptor_t)desc, size);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu: {
        return ascendGetRoPEWorkspaceSize((RoPEAscendDescriptor_t)desc, size);
    }
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaGetRoPEWorkspaceSize((RoPEMacaDescriptor_t)desc, size);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu: {
        return musaGetRoPEWorkspaceSize((RoPEMusaDescriptor_t)desc, size);
    }
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopRoPE(infiniopRoPEDescriptor_t desc,
                                void *workspace, size_t workspace_size,
                                void *t, const void *pos_ids,
                                const void *sin_table, const void *cos_table,
                                void *stream) {
    switch (desc->device_type) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuRoPE((RoPECpuDescriptor_t)desc, workspace, workspace_size, t,
                       pos_ids, sin_table, cos_table, stream);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu: {
        return cudaRoPE((RoPECudaDescriptor_t)desc, workspace, workspace_size,
                        t, pos_ids, sin_table, cos_table, stream);
    }

#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangRoPE((RoPEBangDescriptor_t)desc, workspace, workspace_size,
                        t, pos_ids, sin_table, cos_table, stream);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu: {
        return ascendRoPE((RoPEAscendDescriptor_t)desc, workspace,
                          workspace_size, t, pos_ids, sin_table, cos_table,
                          stream);
    }
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaRoPE((RoPEMacaDescriptor_t)desc, workspace, workspace_size,
                        t, pos_ids, sin_table, cos_table, stream);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu: {
        return musaRoPE((RoPEMusaDescriptor_t)desc, workspace, workspace_size,
                        t, pos_ids, sin_table, cos_table, stream);
    }
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t
infiniopDestroyRoPEDescriptor(infiniopRoPEDescriptor_t desc) {
    switch (desc->device_type) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuDestroyRoPEDescriptor((RoPECpuDescriptor_t)desc);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu: {
        return cudaDestroyRoPEDescriptor((RoPECudaDescriptor_t)desc);
    }

#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangDestroyRoPEDescriptor((RoPEBangDescriptor_t)desc);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu: {
        return ascendDestroyRoPEDescriptor((RoPEAscendDescriptor_t)desc);
    }
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaDestroyRoPEDescriptor((RoPEMacaDescriptor_t)desc);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu: {
        return musaDestroyRoPEDescriptor((RoPEMusaDescriptor_t)desc);
    }
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
