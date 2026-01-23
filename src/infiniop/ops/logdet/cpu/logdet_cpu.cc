#include "logdet_cpu.h"
#include "../../../utils.h"
#include <cmath>
#include <cstring>

namespace op::logdet::cpu {

utils::Result<LogdetInfo> LogdetInfo::create(
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() != 2 || x_shape[0] != x_shape[1]) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Output is scalar
    if (y_shape.size() != 0 && (y_shape.size() != 1 || y_shape[0] != 1)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    LogdetInfo info;
    info.matrix_size = x_shape[0];
    info.input_size = x_desc->numel();

    return utils::Result<LogdetInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    auto info_result = LogdetInfo::create(x_desc, y_desc);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// LU decomposition for computing determinant
template <typename T>
bool lu_decompose(const T *A, T *L, T *U, size_t n) {
    // Initialize L as identity, U as copy of A
    std::memset(L, 0, n * n * sizeof(T));
    std::memcpy(U, A, n * n * sizeof(T));
    for (size_t i = 0; i < n; ++i) {
        L[i * n + i] = utils::cast<T>(1.0);
    }

    for (size_t k = 0; k < n; ++k) {
        if (std::abs(U[k * n + k]) < utils::cast<T>(1e-10)) {
            return false;  // Singular matrix
        }
        for (size_t i = k + 1; i < n; ++i) {
            T factor = U[i * n + k] / U[k * n + k];
            L[i * n + k] = factor;
            for (size_t j = k; j < n; ++j) {
                U[i * n + j] -= factor * U[k * n + j];
            }
        }
    }
    return true;
}

template <typename T>
void logdet_impl(
    const LogdetInfo &info,
    T *y,
    const T *x,
    void *workspace) {

    size_t n = info.matrix_size;
    T *L = reinterpret_cast<T *>(workspace);
    T *U = L + n * n;

    // Perform LU decomposition
    if (!lu_decompose(x, L, U, n)) {
        // Singular matrix: logdet = -inf
        y[0] = utils::cast<T>(-std::numeric_limits<double>::infinity());
        return;
    }

    // Compute log(det) = sum(log(diag(U)))
    T logdet_val = utils::cast<T>(0.0);
    int sign = 1;
    for (size_t i = 0; i < n; ++i) {
        T diag = U[i * n + i];
        if (diag < utils::cast<T>(0.0)) {
            sign *= -1;
            diag = -diag;
        }
        logdet_val += std::log(diag);
    }

    y[0] = logdet_val;
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

    switch (_dtype) {
    case INFINI_DTYPE_F32:
        logdet_impl<float>(_info, reinterpret_cast<float *>(y), reinterpret_cast<const float *>(x), workspace);
        break;
    case INFINI_DTYPE_F64:
        logdet_impl<double>(_info, reinterpret_cast<double *>(y), reinterpret_cast<const double *>(x), workspace);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::logdet::cpu
