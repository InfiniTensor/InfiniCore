#include "spmv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../info.h"
#include <cstring>

namespace op::spmv::cpu {

struct Descriptor::Opaque {
    // CPU doesn't need special hardware context
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    size_t num_cols,
    size_t num_rows,
    size_t nnz,
    infiniDtype_t dtype) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    // 当前仅支持单精度
    if (dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = SpMVInfo::create(num_cols, num_rows, nnz);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(),
        new Opaque{},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// CSR Implementation of SpMV
static void spmv_csr_impl(
    float *y, const float *x, const float *values,
    const int32_t *row_ptr, const int32_t *col_idx,
    size_t num_rows) {

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < static_cast<int>(num_rows); ++i) {
        float sum = 0;
        for (int32_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }
}

infiniStatus_t Descriptor::calculate(
    void *y,
    const void *x,
    const void *values,
    const void *row_ptr,
    const void *col_indices,
    void *stream) const {

    auto validation_result = validateSpMVCSR(
        y, x, values, row_ptr, col_indices, _dtype);
    CHECK_OR_RETURN(validation_result == INFINI_STATUS_SUCCESS, validation_result);

    spmv_csr_impl(
        static_cast<float *>(y),
        static_cast<const float *>(x),
        static_cast<const float *>(values),
        static_cast<const int32_t *>(row_ptr),
        static_cast<const int32_t *>(col_indices),
        _info.num_rows);

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::spmv::cpu
