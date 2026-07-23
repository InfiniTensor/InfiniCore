#ifndef __INFINIOP_MAMBA_SELECTIVE_SCAN_API_H__
#define __INFINIOP_MAMBA_SELECTIVE_SCAN_API_H__

#include "../operator_descriptor.h"
#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

typedef struct InfiniopDescriptor *infiniopMambaSelectiveScanDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateMambaSelectiveScanDescriptor(
    infiniopHandle_t handle,
    infiniopMambaSelectiveScanDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t dt_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_log_desc,
    infiniopTensorDescriptor_t d_desc,
    infiniopTensorDescriptor_t gate_desc,
    infiniopTensorDescriptor_t dt_bias_desc,
    infiniopTensorDescriptor_t state_desc);

__INFINI_C __export infiniStatus_t infiniopGetMambaSelectiveScanWorkspaceSize(
    infiniopMambaSelectiveScanDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopMambaSelectiveScan(
    infiniopMambaSelectiveScanDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *x,
    const void *dt,
    const void *b,
    const void *c,
    const void *a_log,
    const void *d,
    const void *gate,
    const void *dt_bias,
    void *state,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyMambaSelectiveScanDescriptor(
    infiniopMambaSelectiveScanDescriptor_t desc);

#endif
