#include "logdet_moore.h"
#include "../../../utils.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace op::logdet::moore {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() != 2 || x_shape[0] != x_shape[1]) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (y_shape.size() != 0 && (y_shape.size() != 1 || y_shape[0] != 1)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    *desc_ptr = new Descriptor(dtype, x_shape[0], x_desc->numel(),
                               handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    if (workspace_size < this->workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    size_t input_bytes = input_size * infiniopGetDtypeSize(_dtype);
    std::vector<float> h_matrix(input_size);
    CHECK_MOORE(musaMemcpyAsync(h_matrix.data(), x, input_bytes, musaMemcpyDeviceToHost, musa_stream));
    CHECK_MOORE(musaStreamSynchronize(musa_stream));

    std::vector<float> L(matrix_size * matrix_size, 0.0f);
    std::vector<float> U(matrix_size * matrix_size);
    std::memcpy(U.data(), h_matrix.data(), input_bytes);

    for (size_t i = 0; i < matrix_size; ++i) {
        L[i * matrix_size + i] = 1.0f;
    }

    for (size_t k = 0; k < matrix_size; ++k) {
        if (std::abs(U[k * matrix_size + k]) < 1e-10f) {
            if (_dtype == INFINI_DTYPE_F32) {
                *reinterpret_cast<float *>(y) = -std::numeric_limits<float>::infinity();
            } else {
                *reinterpret_cast<double *>(y) = -std::numeric_limits<double>::infinity();
            }
            return INFINI_STATUS_SUCCESS;
        }
        for (size_t i = k + 1; i < matrix_size; ++i) {
            float factor = U[i * matrix_size + k] / U[k * matrix_size + k];
            L[i * matrix_size + k] = factor;
            for (size_t j = k; j < matrix_size; ++j) {
                U[i * matrix_size + j] -= factor * U[k * matrix_size + j];
            }
        }
    }

    float logdet_val = 0.0f;
    for (size_t i = 0; i < matrix_size; ++i) {
        float diag = U[i * matrix_size + i];
        if (diag < 0.0f) diag = -diag;
        logdet_val += std::log(diag);
    }

    if (_dtype == INFINI_DTYPE_F32) {
        CHECK_MOORE(musaMemcpyAsync(y, &logdet_val, sizeof(float), musaMemcpyHostToDevice, musa_stream));
    } else {
        double logdet_val_d = static_cast<double>(logdet_val);
        CHECK_MOORE(musaMemcpyAsync(y, &logdet_val_d, sizeof(double), musaMemcpyHostToDevice, musa_stream));
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::logdet::moore
