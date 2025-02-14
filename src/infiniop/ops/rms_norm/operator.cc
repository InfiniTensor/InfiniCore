#include "infiniop/ops/rms_norm.h"

__C infiniopStatus_t infiniopCreateRMSNormDescriptor(
    infiniopHandle_t handle,
    infiniopRMSNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateRMSNormDescriptor(handle, (RMSNormCpuDescriptor_t *) desc_ptr, y_desc, x_desc, w_desc, epsilon);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateRMSNormDescriptor((infiniopCudaHandle_t) handle, (infiniopRMSNormCudaDescriptor_t *) desc_ptr, y_desc, x_desc, w_desc, epsilon);
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangCreateRMSNormDescriptor((BangHandle_t) handle, (RMSNormBangDescriptor_t *) desc_ptr, y_desc, x_desc, w_desc, epsilon);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnCreateRMSNormDescriptor((AscendHandle_t) handle,
                                                (RMSNormAclnnDescriptor_t *) desc_ptr,
                                                y_desc,
                                                x_desc,
                                                w_desc,
                                                epsilon);
        }
#endif
#ifdef ENABLE_METAX_GPU
        case DevMetaxGpu: {
            return macaCreateRMSNormDescriptor((MacaHandle_t) handle, (RMSNormMacaDescriptor_t *) desc_ptr, y_desc, x_desc, w_desc, epsilon);
        }
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
            return musaCreateRMSNormDescriptor((MusaHandle_t) handle, (RMSNormMusaDescriptor_t *) desc_ptr, y_desc, x_desc, w_desc, epsilon);
        }
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniopStatus_t infiniopGetRMSNormWorkspaceSize(infiniopRMSNormDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetRMSNormWorkspaceSize((RMSNormCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetRMSNormWorkspaceSize((infiniopRMSNormCudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangGetRMSNormWorkspaceSize((RMSNormBangDescriptor_t) desc, size);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnGetRMSNormWorkspaceSize((RMSNormAclnnDescriptor_t) desc,
                                                size);
        }
#endif
#ifdef ENABLE_METAX_GPU
        case DevMetaxGpu: {
            return macaGetRMSNormWorkspaceSize((RMSNormMacaDescriptor_t) desc, size);
        }
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
            return musaGetRMSNormWorkspaceSize((RMSNormMusaDescriptor_t) desc, size);
        }
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniopStatus_t infiniopRMSNorm(infiniopRMSNormDescriptor_t desc, void *workspace, uint64_t workspace_size,
                                     void *y, void const *x, void const *w, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuRMSNorm((RMSNormCpuDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaRMSNorm((infiniopRMSNormCudaDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangRMSNorm((RMSNormBangDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnRMSNorm((RMSNormAclnnDescriptor_t) desc,
                                workspace,
                                workspace_size,
                                y,
                                x,
                                w,
                                stream);
        }
#endif
#ifdef ENABLE_METAX_GPU
        case DevMetaxGpu: {
            return macaRMSNorm((RMSNormMacaDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
        }
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
            return musaRMSNorm((RMSNormMusaDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
        }
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniopStatus_t infiniopDestroyRMSNormDescriptor(infiniopRMSNormDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyRMSNormDescriptor((RMSNormCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyRMSNormDescriptor((infiniopRMSNormCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangDestroyRMSNormDescriptor((RMSNormBangDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnDestroyRMSNormDescriptor((RMSNormAclnnDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_METAX_GPU
        case DevMetaxGpu: {
            return macaDestroyRMSNormDescriptor((RMSNormMacaDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
            return musaDestroyRMSNormDescriptor((RMSNormMusaDescriptor_t) desc);
        }
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
