#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/mamba2_scan.h"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/mamba2_scan_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/mamba2_scan_metax.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateMamba2ScanDescriptor(
    infiniopHandle_t handle, infiniopMamba2ScanDescriptor_t *desc_ptr, infiniopTensorDescriptor_t out_desc, infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t dt_desc, infiniopTensorDescriptor_t b_desc, infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t d_desc, infiniopTensorDescriptor_t dt_bias_desc, infiniopTensorDescriptor_t state_desc, infiniopTensorDescriptor_t offsets_desc, infiniopTensorDescriptor_t initial_indices_desc, infiniopTensorDescriptor_t final_indices_desc) {
    if (handle == nullptr || desc_ptr == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return op::mamba2_scan::nvidia::Descriptor::create(handle, reinterpret_cast<op::mamba2_scan::nvidia::Descriptor **>(desc_ptr), out_desc, x_desc, dt_desc, b_desc, c_desc, a_desc, d_desc, dt_bias_desc, state_desc, offsets_desc, initial_indices_desc, final_indices_desc);
#endif
#ifdef ENABLE_METAX_API
    case INFINI_DEVICE_METAX:
        return op::mamba2_scan::metax::Descriptor::create(handle, reinterpret_cast<op::mamba2_scan::metax::Descriptor **>(desc_ptr), out_desc, x_desc, dt_desc, b_desc, c_desc, a_desc, d_desc, dt_bias_desc, state_desc, offsets_desc, initial_indices_desc, final_indices_desc);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
__INFINI_C infiniStatus_t infiniopGetMamba2ScanWorkspaceSize(infiniopMamba2ScanDescriptor_t desc, size_t *size) {
    if (desc == nullptr || size == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        *size = reinterpret_cast<op::mamba2_scan::nvidia::Descriptor *>(desc)->workspaceSize();
        return INFINI_STATUS_SUCCESS;
#endif
#ifdef ENABLE_METAX_API
    case INFINI_DEVICE_METAX:
        *size = reinterpret_cast<op::mamba2_scan::metax::Descriptor *>(desc)->workspaceSize();
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
__INFINI_C infiniStatus_t infiniopMamba2Scan(infiniopMamba2ScanDescriptor_t desc, void *workspace, size_t workspace_size, void *out, const void *x, const void *dt, const void *b, const void *c, const void *a, const void *d, const void *dt_bias, void *state, const void *offsets, const void *initial_indices, const void *final_indices, void *stream) {
    if (desc == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return reinterpret_cast<op::mamba2_scan::nvidia::Descriptor *>(desc)->calculate(workspace, workspace_size, out, x, dt, b, c, a, d, dt_bias, state, offsets, initial_indices, final_indices, stream);
#endif
#ifdef ENABLE_METAX_API
    case INFINI_DEVICE_METAX:
        return reinterpret_cast<op::mamba2_scan::metax::Descriptor *>(desc)->calculate(workspace, workspace_size, out, x, dt, b, c, a, d, dt_bias, state, offsets, initial_indices, final_indices, stream);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
__INFINI_C infiniStatus_t infiniopDestroyMamba2ScanDescriptor(infiniopMamba2ScanDescriptor_t desc) {
    if (desc == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        delete reinterpret_cast<op::mamba2_scan::nvidia::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
#endif
#ifdef ENABLE_METAX_API
    case INFINI_DEVICE_METAX:
        delete reinterpret_cast<op::mamba2_scan::metax::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
