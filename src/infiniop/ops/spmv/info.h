#ifndef __SPMV_INFO_H__
#define __SPMV_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"

namespace op::spmv {

// SpMV operation information
class SpMVInfo {
    SpMVInfo() = default;

public:
    size_t num_rows;
    size_t num_cols;
    size_t nnz;

    static utils::Result<SpMVInfo> create(
        size_t num_cols,
        size_t num_rows,
        size_t nnz) {

        CHECK_OR_RETURN(num_cols > 0 && num_rows > 0 && nnz > 0,
                        INFINI_STATUS_BAD_PARAM);

        SpMVInfo info;
        info.num_rows = num_rows;
        info.num_cols = num_cols;
        info.nnz = nnz;

        return utils::Result<SpMVInfo>(info);
    }

    static utils::Result<SpMVInfo> createLegacy(
        size_t num_rows,
        size_t num_cols,
        size_t nnz) {

        CHECK_OR_RETURN(num_rows > 0 && num_cols > 0 && nnz > 0,
                        INFINI_STATUS_BAD_PARAM);

        SpMVInfo info;
        info.num_rows = num_rows;
        info.num_cols = num_cols;
        info.nnz = nnz;

        return utils::Result<SpMVInfo>(info);
    }
};

// validate SpMV CSR operation parameters
inline infiniStatus_t validateSpMVCSR(
    const void *y, const void *x, const void *values,
    const void *row_indices, const void *col_indices,
    infiniDtype_t dtype) {

    CHECK_OR_RETURN(y && x && values && row_indices && col_indices,
                    INFINI_STATUS_BAD_PARAM);
    CHECK_OR_RETURN(dtype == INFINI_DTYPE_F32, INFINI_STATUS_BAD_TENSOR_DTYPE);

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::spmv

#endif // __SPMV_INFO_H__
