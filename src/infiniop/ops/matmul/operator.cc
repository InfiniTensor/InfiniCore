#include "../utils.h"
#include "infiniop/ops/matmul.h"

#ifdef ENABLE_CPU_API
#include "cpu/matmul_cpu.h"
#endif
#ifdef ENABLE_CUDA_API
#include "cuda/matmul_cuda_api.h"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/matmul_cnnl.h"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/matmul_aclnn_api.h"
#endif

__C infiniopStatus_t infiniopCreateMatmulDescriptor(infiniopHandle_t handle,
                                                    infiniopMatmulDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t c_desc,
                                                    infiniopTensorDescriptor_t a_desc,
                                                    infiniopTensorDescriptor_t b_desc) {
    switch (handle->device) {
#ifdef ENABLE_CPU_API
        case INFINI_DEVICE_CPU:
            return cpuCreateMatmulDescriptor((infiniopCpuHandle_t) handle, (MatmulCpuDescriptor_t *) desc_ptr, c_desc, a_desc, b_desc);
#endif
#ifdef ENABLE_CUDA_API
        case INFINI_DEVICE_NVIDIA: {
            return cudaCreateMatmulDescriptor((infiniopCudaHandle_t) handle, (infiniopMatmulCudaDescriptor_t *) desc_ptr, c_desc, a_desc, b_desc);
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangCreateMatmulDescriptor((BangHandle_t) handle, (MatmulBangDescriptor_t *) desc_ptr, c_desc, a_desc, b_desc);
        }
#endif
#ifdef ENABLE_ASCEND_API
        case INFINI_DEVICE_ASCEND: {
            return aclnnCreateMatmulDescriptor(
                (infiniopAscendHandle_t)handle,
                (MatmulAclnnDescriptor_t *)desc_ptr, c_desc, a_desc, b_desc, 1);
        }
#endif
    }
    return INFINIOP_STATUS_BAD_DEVICE;
}

__C infiniopStatus_t
infiniopGetMatmulWorkspaceSize(infiniopMatmulDescriptor_t desc, size_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU_API
        case INFINI_DEVICE_CPU:
            return cpuGetMatmulWorkspaceSize((MatmulCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_CUDA_API
        case INFINI_DEVICE_NVIDIA: {
            return cudaGetMatmulWorkspaceSize((infiniopMatmulCudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangGetMatmulWorkspaceSize((MatmulBangDescriptor_t) desc, size);
        }
#endif
#ifdef ENABLE_ASCEND_API
        case INFINI_DEVICE_ASCEND: {
            return aclnnGetMatmulWorkspaceSize((MatmulAclnnDescriptor_t) desc,
                                               size);
        }
#endif
    }
    return INFINIOP_STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopMatmul(infiniopMatmulDescriptor_t desc,
                                    void *workspace, size_t workspace_size,
                                    void *c, void const *a, void const *b,
                                    float alpha, float beta, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU_API
        case INFINI_DEVICE_CPU:
            return cpuMatmul((MatmulCpuDescriptor_t) desc, workspace, workspace_size, c, a, b, alpha, beta);
#endif
#ifdef ENABLE_CUDA_API
        case INFINI_DEVICE_NVIDIA:
            return cudaMatmul((infiniopMatmulCudaDescriptor_t) desc, workspace, workspace_size, c, a, b, alpha, beta, stream);
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangMatmul((MatmulBangDescriptor_t)desc, workspace,
                              workspace_size, c, a, b, alpha, beta, stream);
        }
#endif
#ifdef ENABLE_ASCEND_API
        case INFINI_DEVICE_ASCEND:
            return aclnnMatmul((MatmulAclnnDescriptor_t)desc, workspace,
                               workspace_size, c, a, b, alpha, beta, stream);
#endif
    }
    return INFINIOP_STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyMatmulDescriptor(infiniopMatmulDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU_API
        case INFINI_DEVICE_CPU:
            return cpuDestroyMatmulDescriptor((MatmulCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_CUDA_API
        case INFINI_DEVICE_NVIDIA: {
            return cudaDestroyMatmulDescriptor((infiniopMatmulCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangDestroyMatmulDescriptor((MatmulBangDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_ASCEND_API
        case INFINI_DEVICE_ASCEND: {
            return aclnnDestroyMatmulDescriptor((MatmulAclnnDescriptor_t) desc);
        }
#endif
    }
    return INFINIOP_STATUS_BAD_DEVICE;
}
