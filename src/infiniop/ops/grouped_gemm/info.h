#ifndef __GROUPED_GEMM_INFO_H__
#define __GROUPED_GEMM_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::grouped_gemm {

// Shape/stride metadata captured at descriptor-creation time.
//
// We require the leading-axis layouts that match `torch.matmul` on row-major
// tensors so we can address each group's slab as `base + offset * row_stride`.
struct GroupedGemmInfo {
    infiniDtype_t dtype;
    size_t num_groups;
    size_t m_total;
    size_t n;
    size_t k;

    // Strides in elements (not bytes) on the row/col axes.
    ptrdiff_t a_row_stride; // stride between A rows
    ptrdiff_t a_col_stride; // == 1 expected
    ptrdiff_t c_row_stride; // stride between C rows
    ptrdiff_t c_col_stride; // == 1 expected
    ptrdiff_t b_group_stride; // stride between expert slabs of B
    ptrdiff_t b_row_stride;   // stride between rows of B[g]
    ptrdiff_t b_col_stride;   // == 1 expected

    static utils::Result<GroupedGemmInfo> create(
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc,
        infiniopTensorDescriptor_t group_sizes_desc) {

        if (a_desc->ndim() != 2 || c_desc->ndim() != 2 || b_desc->ndim() != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (group_sizes_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (group_sizes_desc->dtype() != INFINI_DTYPE_I32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        auto dtype = c_desc->dtype();
        if (dtype != a_desc->dtype() || dtype != b_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        size_t m_total = a_desc->dim(0);
        size_t k_a = a_desc->dim(1);
        size_t num_groups = b_desc->dim(0);
        size_t n_b = b_desc->dim(1);
        size_t k_b = b_desc->dim(2);
        size_t m_c = c_desc->dim(0);
        size_t n_c = c_desc->dim(1);

        if (k_a != k_b || m_total != m_c || n_b != n_c) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (group_sizes_desc->dim(0) != num_groups) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // We need the inner (K, N) axes to be unit-strided so each slab is a
        // dense matrix that cuBLAS / our cpu loops can consume directly.
        if (a_desc->stride(1) != 1 || b_desc->stride(2) != 1 || c_desc->stride(1) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        GroupedGemmInfo info;
        info.dtype = dtype;
        info.num_groups = num_groups;
        info.m_total = m_total;
        info.n = n_b;
        info.k = k_a;
        info.a_row_stride = a_desc->stride(0);
        info.a_col_stride = a_desc->stride(1);
        info.c_row_stride = c_desc->stride(0);
        info.c_col_stride = c_desc->stride(1);
        info.b_group_stride = b_desc->stride(0);
        info.b_row_stride = b_desc->stride(1);
        info.b_col_stride = b_desc->stride(2);
        return utils::Result<GroupedGemmInfo>(info);
    }
};

} // namespace op::grouped_gemm

#endif // __GROUPED_GEMM_INFO_H__
