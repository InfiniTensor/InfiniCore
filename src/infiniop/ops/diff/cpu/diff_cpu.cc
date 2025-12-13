#include "diff_cpu.h"
#include "../../../utils.h"
#include <algorithm>
#include <cmath>

namespace op::diff::cpu {

utils::Result<DiffInfo> DiffInfo::create(
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    int dim,
    int n) {

    if (n <= 0) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();
    size_t ndim = x_desc->ndim();

    if (dim < 0) {
        dim += static_cast<int>(ndim);
    }
    if (dim < 0 || dim >= static_cast<int>(ndim)) {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (x_shape[dim] <= static_cast<size_t>(n)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Calculate output shape
    std::vector<size_t> expected_output_shape = x_shape;
    expected_output_shape[dim] -= n;

    if (y_shape != expected_output_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    DiffInfo info;
    info.ndim = ndim;
    info.dim = dim;
    info.n = n;
    info.input_shape = x_shape;
    info.output_shape = y_shape;
    info.input_strides = x_desc->strides();
    info.output_strides = y_desc->strides();
    info.input_size = x_desc->numel();
    info.output_size = y_desc->numel();

    return utils::Result<DiffInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int dim,
    int n) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto info_result = DiffInfo::create(x_desc, y_desc, dim, n);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void diff_impl(
    const DiffInfo &info,
    T *y,
    const T *x) {

    // Compute n-th order difference along specified dimension
    // For n=1: y[i] = x[i+1] - x[i]
    // For n>1: recursively apply diff

    size_t dim_size = info.input_shape[info.dim];
    size_t output_dim_size = info.output_shape[info.dim];

    // Calculate sizes before and after the dimension
    size_t size_before = 1;
    for (size_t i = 0; i < static_cast<size_t>(info.dim); ++i) {
        size_before *= info.input_shape[i];
    }
    size_t size_after = 1;
    for (size_t i = static_cast<size_t>(info.dim) + 1; i < info.ndim; ++i) {
        size_after *= info.input_shape[i];
    }

    // Allocate temporary buffer for recursive diff computation
    std::vector<T> temp_input(info.input_size);
    std::vector<T> temp_output(info.output_size);
    std::memcpy(temp_input.data(), x, info.input_size * sizeof(T));

    // Apply diff n times
    for (int order = 0; order < info.n; ++order) {
        size_t current_dim_size = dim_size - order;
        size_t current_output_size = current_dim_size - 1;

#pragma omp parallel for collapse(2)
        for (ptrdiff_t b = 0; b < static_cast<ptrdiff_t>(size_before); ++b) {
            for (ptrdiff_t a = 0; a < static_cast<ptrdiff_t>(size_after); ++a) {
                for (size_t i = 0; i < current_output_size; ++i) {
                    size_t idx1 = b * current_dim_size * size_after + i * size_after + a;
                    size_t idx2 = b * current_dim_size * size_after + (i + 1) * size_after + a;
                    size_t out_idx = b * current_output_size * size_after + i * size_after + a;
                    temp_output[out_idx] = temp_input[idx2] - temp_input[idx1];
                }
            }
        }

        if (order < info.n - 1) {
            std::swap(temp_input, temp_output);
            current_dim_size = current_output_size;
        }
    }

    // Copy final result to output
    std::memcpy(y, temp_output.data(), info.output_size * sizeof(T));
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        diff_impl<fp16_t>(_info, reinterpret_cast<fp16_t *>(y), reinterpret_cast<const fp16_t *>(x));
        break;
    case INFINI_DTYPE_BF16:
        diff_impl<bf16_t>(_info, reinterpret_cast<bf16_t *>(y), reinterpret_cast<const bf16_t *>(x));
        break;
    case INFINI_DTYPE_F32:
        diff_impl<float>(_info, reinterpret_cast<float *>(y), reinterpret_cast<const float *>(x));
        break;
    case INFINI_DTYPE_F64:
        diff_impl<double>(_info, reinterpret_cast<double *>(y), reinterpret_cast<const double *>(x));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::diff::cpu
