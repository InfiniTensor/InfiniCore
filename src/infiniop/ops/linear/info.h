#ifndef __GEMM_INFO_H__
#define __GEMM_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <algorithm>

namespace op::linear {

struct BlasMatrix {
    size_t ndim;
    size_t batch;
    ptrdiff_t stride;
    size_t rows;
    size_t cols;
    ptrdiff_t row_stride;
    ptrdiff_t col_stride;
    infiniDtype_t dtype;

    static utils::Result<BlasMatrix> create(infiniopTensorDescriptor_t layout) {
        BlasMatrix ans;

        if (layout->ndim() == 2) {
            ans.ndim = 2;
            ans.batch = 1;
            ans.stride = 0;
            ans.rows = layout->dim(0);
            ans.cols = layout->dim(1);
            ans.row_stride = layout->stride(0);
            ans.col_stride = layout->stride(1);
            ans.dtype = layout->dtype();
        } else if (layout->ndim() == 3) {
            ans.ndim = 3;
            ans.batch = layout->dim(0);
            ans.stride = ans.batch == 1 ? 0 : layout->stride(0);
            ans.rows = layout->dim(1);
            ans.cols = layout->dim(2);
            ans.row_stride = layout->stride(1);
            ans.col_stride = layout->stride(2);
            ans.dtype = layout->dtype();
        } else {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (ans.row_stride != 1 && ans.col_stride != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        return utils::Result<BlasMatrix>(ans);
    }

    bool match_batch(size_t _batch) const {
        return batch == _batch || batch == 1;
    }

    void transpose() {
        std::swap(rows, cols);
        std::swap(row_stride, col_stride);
    }

    ptrdiff_t ld() const { return row_stride == 1 ? col_stride : row_stride; }
};

enum class MatrixLayout : char {
    COL_MAJOR,
    ROW_MAJOR,
};

class MatmulInfo {
    MatmulInfo() = default;

public:
    BlasMatrix a_matrix;
    BlasMatrix b_matrix;
    BlasMatrix c_matrix;
    BlasMatrix d_matrix;

    size_t m, n, k, batch;
    bool is_transed;

    static utils::Result<MatmulInfo> create(infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc,
                                            infiniopTensorDescriptor_t c_desc,
                                            infiniopTensorDescriptor_t d_desc,
                                            MatrixLayout layout) {

        auto a_matrix = BlasMatrix::create(a_desc);
        CHECK_RESULT(a_matrix);

        auto b_matrix = BlasMatrix::create(b_desc);
        CHECK_RESULT(b_matrix);

        auto c_matrix = BlasMatrix::create(c_desc);
        CHECK_RESULT(c_matrix);

        auto d_matrix = BlasMatrix::create(d_desc);
        CHECK_RESULT(d_matrix);

        if ((c_matrix->rows != a_matrix->rows || c_matrix->cols != b_matrix->cols || a_matrix->cols != b_matrix->rows) && a_desc->dtype() != INFINI_DTYPE_F8_E4M3 && a_desc->dtype() != INFINI_DTYPE_F8_E5M2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto batch = c_matrix->batch;
        if (!a_matrix->match_batch(batch) || !b_matrix->match_batch(batch)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto is_transed = false;
        if (((layout == MatrixLayout::COL_MAJOR && c_matrix->col_stride == 1) || (layout == MatrixLayout::ROW_MAJOR && c_matrix->row_stride == 1)) && a_desc->dtype() != INFINI_DTYPE_F8_E4M3 && a_desc->dtype() != INFINI_DTYPE_F8_E5M2) {
            c_matrix->transpose();
            b_matrix->transpose();
            a_matrix->transpose();
            std::swap(a_matrix, b_matrix);
            is_transed = true;
        }

        auto m = c_matrix->rows;
        auto n = c_matrix->cols;
        auto k = a_matrix->cols;

        return utils::Result<MatmulInfo>(
            MatmulInfo{a_matrix.take(), b_matrix.take(), c_matrix.take(),
                       d_matrix.take(), m, n, k, batch, is_transed});
    }
};

} // namespace op::linear

#endif // __GEMM_INFO_H__
