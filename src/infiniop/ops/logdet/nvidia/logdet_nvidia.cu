#include "logdet_nvidia.cuh"
#include "../../../utils.h"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace op::logdet::nvidia {

Descriptor::~Descriptor() = default;

template <typename T>
__global__ void pack_matrix_kernel(
    T *dst,
    const T *src,
    ptrdiff_t s0,
    ptrdiff_t s1,
    size_t n) {

    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = n * n;
    if (idx >= total) {
        return;
    }
    const size_t i = idx / n;
    const size_t j = idx % n;
    dst[idx] = src[static_cast<ptrdiff_t>(i) * s0 + static_cast<ptrdiff_t>(j) * s1];
}

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

    *desc_ptr = new Descriptor(dtype, x_shape[0], x_desc->numel(), x_desc->strides(),
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

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    auto run_host_lu = [&](auto tag) -> infiniStatus_t {
        using T = decltype(tag);
        const size_t input_bytes = input_size * sizeof(T);
        T *packed = reinterpret_cast<T *>(workspace);
        const ptrdiff_t s0 = input_strides[0];
        const ptrdiff_t s1 = input_strides[1];

        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((input_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        pack_matrix_kernel<T><<<blocks, BLOCK_SIZE, 0, cuda_stream>>>(packed, reinterpret_cast<const T *>(x), s0, s1, matrix_size);

        std::vector<T> h_matrix(input_size);
        CHECK_CUDA(cudaMemcpyAsync(h_matrix.data(), packed, input_bytes, cudaMemcpyDeviceToHost, cuda_stream));
        CHECK_CUDA(cudaStreamSynchronize(cuda_stream));

        // In-place LU decomposition on host (with partial pivoting) to compute sign + log|det|.
        std::vector<T> U = std::move(h_matrix);
        int det_sign = 1;
        double log_abs_det = 0.0;
        const double eps = std::is_same_v<T, float> ? 1e-6 : 1e-12;

        for (size_t k = 0; k < matrix_size; ++k) {
            size_t pivot_row = k;
            double pivot_abs = std::abs(static_cast<double>(U[k * matrix_size + k]));
            for (size_t i = k + 1; i < matrix_size; ++i) {
                const double v = std::abs(static_cast<double>(U[i * matrix_size + k]));
                if (v > pivot_abs) {
                    pivot_abs = v;
                    pivot_row = i;
                }
            }

            if (pivot_abs <= eps) {
                const T neg_inf = -std::numeric_limits<T>::infinity();
                CHECK_CUDA(cudaMemcpyAsync(y, &neg_inf, sizeof(T), cudaMemcpyHostToDevice, cuda_stream));
                return INFINI_STATUS_SUCCESS;
            }

            if (pivot_row != k) {
                for (size_t j = 0; j < matrix_size; ++j) {
                    std::swap(U[k * matrix_size + j], U[pivot_row * matrix_size + j]);
                }
                det_sign *= -1;
            }

            const T pivot = U[k * matrix_size + k];
            if (pivot < static_cast<T>(0)) {
                det_sign *= -1;
            }
            log_abs_det += std::log(std::abs(static_cast<double>(pivot)));

            for (size_t i = k + 1; i < matrix_size; ++i) {
                const T factor = U[i * matrix_size + k] / pivot;
                U[i * matrix_size + k] = static_cast<T>(0);
                for (size_t j = k + 1; j < matrix_size; ++j) {
                    U[i * matrix_size + j] -= factor * U[k * matrix_size + j];
                }
            }
        }

        const T out =
            (det_sign <= 0)
                ? static_cast<T>(std::numeric_limits<double>::quiet_NaN())
                : static_cast<T>(log_abs_det);
        CHECK_CUDA(cudaMemcpyAsync(y, &out, sizeof(T), cudaMemcpyHostToDevice, cuda_stream));
        return INFINI_STATUS_SUCCESS;
    };

    if (_dtype == INFINI_DTYPE_F32) {
        return run_host_lu(float{});
    }
    return run_host_lu(double{});
}

} // namespace op::logdet::nvidia
