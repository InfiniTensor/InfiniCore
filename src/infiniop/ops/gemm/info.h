#ifndef __GEMM_INFO_H__
#define __GEMM_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <algorithm>
#include <utility> 

namespace op::gemm {

class BlasMatrix {
public: 
    BlasMatrix() = default;

    size_t ndim;
    size_t batch;
    ptrdiff_t stride;
    size_t rows;
    size_t cols;
    ptrdiff_t row_stride;
    ptrdiff_t col_stride;

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
        } else if (layout->ndim() == 3) {
            ans.ndim = 3;
            ans.batch = layout->dim(0);
            ans.stride = ans.batch == 1 ? 0 : layout->stride(0);
            ans.rows = layout->dim(1);
            ans.cols = layout->dim(2);
            ans.row_stride = layout->stride(1);
            ans.col_stride = layout->stride(2);
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

    ptrdiff_t ld() const {
        return row_stride == 1 ? col_stride : row_stride;
    }
};

enum class MatrixLayout : char {
    COL_MAJOR,
    ROW_MAJOR,
};

class MatmulInfo {
public: 
    MatmulInfo() = default;

    BlasMatrix a_matrix;
    BlasMatrix b_matrix;
    BlasMatrix c_matrix;

    size_t m, n, k, batch;
    bool is_transed;

private: 
    MatmulInfo(BlasMatrix a, BlasMatrix b, BlasMatrix c, size_t m_val, size_t n_val, size_t k_val, size_t batch_val, bool transed)
        : a_matrix(std::move(a)),
          b_matrix(std::move(b)),
          c_matrix(std::move(c)),
          m(m_val),
          n(n_val),
          k(k_val),
          batch(batch_val),
          is_transed(transed) {}

public:
    static utils::Result<MatmulInfo> create(
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc,
        MatrixLayout layout) {

        auto a_matrix_res = BlasMatrix::create(a_desc);
        CHECK_RESULT(a_matrix_res);
        BlasMatrix a_matrix = a_matrix_res.take(); 

        auto b_matrix_res = BlasMatrix::create(b_desc);
        CHECK_RESULT(b_matrix_res);
        BlasMatrix b_matrix = b_matrix_res.take(); 

        auto c_matrix_res = BlasMatrix::create(c_desc);
        CHECK_RESULT(c_matrix_res);
        BlasMatrix c_matrix = c_matrix_res.take(); 

     
        if (c_matrix.rows != a_matrix.rows || c_matrix.cols != b_matrix.cols || a_matrix.cols != b_matrix.rows) {
             return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto batch_val = c_matrix.batch; 
        if (!a_matrix.match_batch(batch_val) || !b_matrix.match_batch(batch_val)) {
             return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto is_transed_val = false;
        if ((layout == MatrixLayout::COL_MAJOR && c_matrix.col_stride == 1)
            || (layout == MatrixLayout::ROW_MAJOR && c_matrix.row_stride == 1)) {
            c_matrix.transpose();
            b_matrix.transpose();
            a_matrix.transpose();
            std::swap(a_matrix, b_matrix); 
            is_transed_val = true;
        }

        auto m_val = c_matrix.rows; 
        auto n_val = c_matrix.cols; 
        auto k_val = a_matrix.cols; 


        return utils::Result<MatmulInfo>(MatmulInfo{
            std::move(a_matrix), 
            std::move(b_matrix),
            std::move(c_matrix),
            m_val,
            n_val,
            k_val,
            batch_val,
            is_transed_val});
    }
};

} // namespace op::gemm

#endif // __GEMM_INFO_H__