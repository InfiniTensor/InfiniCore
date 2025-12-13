#include "matrix_power_moore.h"
#include "../../../utils.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstring>

namespace op::matrix_power::moore {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int n) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() != 2 || x_shape[0] != x_shape[1]) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (y_shape != x_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    *desc_ptr = new Descriptor(dtype, x_shape[0], (n < 0) ? -n : n,
                               x_desc->numel(), y_desc->numel(),
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

    std::vector<float> result(output_size, 0.0f);
    std::vector<float> temp1(input_size);
    std::vector<float> temp2(input_size);
    std::memcpy(temp1.data(), h_matrix.data(), input_bytes);

    for (size_t i = 0; i < matrix_size; ++i) {
        result[i * matrix_size + i] = 1.0f;
    }

    int power = static_cast<int>(n);
    while (power > 0) {
        if (power & 1) {
            std::fill(temp2.begin(), temp2.end(), 0.0f);
            for (size_t i = 0; i < matrix_size; ++i) {
                for (size_t k = 0; k < matrix_size; ++k) {
                    float val = result[i * matrix_size + k];
                    for (size_t j = 0; j < matrix_size; ++j) {
                        temp2[i * matrix_size + j] += val * temp1[k * matrix_size + j];
                    }
                }
            }
            std::memcpy(result.data(), temp2.data(), output_size * sizeof(float));
        }
        std::fill(temp2.begin(), temp2.end(), 0.0f);
        for (size_t i = 0; i < matrix_size; ++i) {
            for (size_t k = 0; k < matrix_size; ++k) {
                float val = temp1[i * matrix_size + k];
                for (size_t j = 0; j < matrix_size; ++j) {
                    temp2[i * matrix_size + j] += val * temp1[k * matrix_size + j];
                }
            }
        }
        std::memcpy(temp1.data(), temp2.data(), input_bytes);
        power >>= 1;
    }

    CHECK_MOORE(musaMemcpyAsync(y, result.data(), input_bytes, musaMemcpyHostToDevice, musa_stream));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::matrix_power::moore
