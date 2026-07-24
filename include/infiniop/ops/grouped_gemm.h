#ifndef __INFINIOP_GROUPED_GEMM_API_H__
#define __INFINIOP_GROUPED_GEMM_API_H__

#include "../operator_descriptor.h"

/**
 * Variable-batched (a.k.a. "grouped" or "segment") GEMM.
 *
 * For each group `g` in `[0, num_groups)`:
 *   rows_g  = group_sizes[g]
 *   off_g   = sum(group_sizes[0..g])
 *   A_g     = a[off_g : off_g + rows_g, :]            // [rows_g, K]
 *   B_g     = b[g, :, :]                              // [N, K]
 *   c[off_g : off_g + rows_g, :] = alpha * A_g @ B_g^T + beta * c_g
 *
 * Shapes
 *   a : [M_total, K]
 *   b : [num_groups, N, K]   -- matches PyTorch / HF linear weight layout
 *   c : [M_total, N]
 *   group_sizes : [num_groups], int32, sum == M_total
 *
 * Constraints
 *   - M_total = sum(group_sizes); enforced at calculate-time.
 *   - All `*_desc` must be row-major contiguous on the leading axes.
 *   - dtype: F16, BF16 or F32 (same for a, b, c).
 *   - `group_sizes` is `INFINI_DTYPE_I32`.
 *   - The `group_sizes` data pointer lives on the same device as `a/b/c`.
 *     CPU backends read it directly; device backends sync it to host
 *     inside `calculate`.
 *   - `group_sizes_host` (optional, may be NULL): host-side copy of the same
 *     int32 group sizes. When provided, device backends use it directly and
 *     skip the per-call device->host copy + stream sync. The caller must keep
 *     it valid and consistent with `group_sizes` until the call returns. NULL
 *     preserves the legacy device-sync behavior. Unsafe under graph capture
 *     (pass NULL there).
 */
typedef struct InfiniopDescriptor *infiniopGroupedGemmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateGroupedGemmDescriptor(
    infiniopHandle_t handle,
    infiniopGroupedGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t group_sizes_desc);

__INFINI_C __export infiniStatus_t infiniopGetGroupedGemmWorkspaceSize(
    infiniopGroupedGemmDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopGroupedGemm(
    infiniopGroupedGemmDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    const void *group_sizes,
    const void *group_sizes_host,
    float alpha,
    float beta,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyGroupedGemmDescriptor(
    infiniopGroupedGemmDescriptor_t desc);

#endif // __INFINIOP_GROUPED_GEMM_API_H__
