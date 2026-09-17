#ifndef INFINIOP_MAMBA2_SCAN_API_H_
#define INFINIOP_MAMBA2_SCAN_API_H_
#include "../operator_descriptor.h"
#include <stddef.h>

typedef struct InfiniopDescriptor *infiniopMamba2ScanDescriptor_t;

// All tensors are contiguous. `out`/`x`: [tokens, heads, head_dim], `dt`:
// [tokens, heads], `b`/`c`: [tokens, groups, state_size], sharing F32/F16/BF16.
// `a`/`d`/`dt_bias`: FP32 [heads]. `a` contains the transformed -exp(A_log).
// `state`: FP32 [pool, heads, head_dim, state_size], with state_size <= 256.
// `offsets`: int32 [requests + 1], strictly increasing from zero to tokens.
// Initial/final indices are int32 [requests] and must name valid pool rows.
// Final rows are unique and nonzero. A request may update its own initial row;
// no request may read another request's final row. Row zero is read-only.
// Output, inputs, state, and workspace must not overlap in storage. Metadata
// values are caller-validated preconditions; execution does not synchronize
// the device to inspect them on the host.
__INFINI_C __export infiniStatus_t infiniopCreateMamba2ScanDescriptor(
    infiniopHandle_t handle, infiniopMamba2ScanDescriptor_t *desc_ptr, infiniopTensorDescriptor_t out_desc, infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t dt_desc, infiniopTensorDescriptor_t b_desc, infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t d_desc, infiniopTensorDescriptor_t dt_bias_desc, infiniopTensorDescriptor_t state_desc, infiniopTensorDescriptor_t offsets_desc, infiniopTensorDescriptor_t initial_indices_desc, infiniopTensorDescriptor_t final_indices_desc);
__INFINI_C __export infiniStatus_t infiniopGetMamba2ScanWorkspaceSize(infiniopMamba2ScanDescriptor_t desc, size_t *size);
__INFINI_C __export infiniStatus_t infiniopMamba2Scan(
    infiniopMamba2ScanDescriptor_t desc, void *workspace, size_t workspace_size, void *out, const void *x, const void *dt, const void *b, const void *c, const void *a, const void *d, const void *dt_bias, void *state, const void *offsets, const void *initial_indices, const void *final_indices, void *stream);
__INFINI_C __export infiniStatus_t infiniopDestroyMamba2ScanDescriptor(infiniopMamba2ScanDescriptor_t desc);
#endif
