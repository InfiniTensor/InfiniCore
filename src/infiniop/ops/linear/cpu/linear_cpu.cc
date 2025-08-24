#include "linear_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../tensor.h"
#include "../../../../utils/custom_types.h"

namespace op::linear::cpu {

// Linear kernel implementation
template<typename T>
static infiniStatus_t linear_kernel(
    T *y,
    const T *x,
    const T *w,
    const T *b,
    const std::vector<int> &x_dims,
    const std::vector<int> &w_dims) {

    int batch_size = 1;
    for (size_t i = 0; i < x_dims.size() - 1; i++) {
        batch_size *= x_dims[i];
    }
    int in_features = x_dims[x_dims.size() - 1];
    int out_features = w_dims[0];
    
    // Perform y = x * w^T + b
    for (int batch = 0; batch < batch_size; batch++) {
        for (int out = 0; out < out_features; out++) {
            if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
                float sum = 0.0f;
                for (int in = 0; in < in_features; in++) {
                    sum += utils::cast<float>(x[batch * in_features + in]) * utils::cast<float>(w[out * in_features + in]);
                }
                if (b != nullptr) {
                    sum += utils::cast<float>(b[out]);
                }
                y[batch * out_features + out] = utils::cast<T>(sum);
            } else {
                T sum{};
                for (int in = 0; in < in_features; in++) {
                    sum += x[batch * in_features + in] * w[out * in_features + in];
                }
                if (b != nullptr) {
                    sum += b[out];
                }
                y[batch * out_features + out] = sum;
            }
        }
    }
    
    return INFINI_STATUS_SUCCESS;
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t y_desc) {

    // Check device type
    if (handle->device != INFINI_DEVICE_CPU) {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

    // Check data types
    auto x_dtype = x_desc->dtype();
    auto w_dtype = w_desc->dtype();
    auto y_dtype = y_desc->dtype();

    if (x_dtype != w_dtype || x_dtype != y_dtype) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // Check bias data type if provided
    if (b_desc) {
        auto b_dtype = b_desc->dtype();
        if (b_dtype != x_dtype) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }

    // Check dimensions
    auto x_shape = x_desc->shape();
    auto w_shape = w_desc->shape();
    auto y_shape = y_desc->shape();
    
    int x_ndim = x_shape.size();
    int w_ndim = w_shape.size();
    int y_ndim = y_shape.size();

    if (w_ndim != 2 || x_ndim < 1 || y_ndim < 1) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // Get dimensions
    std::vector<int> x_dims(x_shape.begin(), x_shape.end());
    std::vector<int> w_dims(w_shape.begin(), w_shape.end());
    std::vector<int> y_dims(y_shape.begin(), y_shape.end());

    // Check dimension compatibility
    // x: (..., in_features), w: (out_features, in_features), y: (..., out_features)
    int in_features = x_dims[x_ndim - 1];
    int out_features = w_dims[0];
    int w_in_features = w_dims[1];

    if (in_features != w_in_features) {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (y_dims[y_ndim - 1] != out_features) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // Check bias dimensions if provided
    if (b_desc) {
        auto b_shape = b_desc->shape();
        int b_ndim = b_shape.size();
        std::vector<int> b_dims(b_shape.begin(), b_shape.end());

        if (b_ndim != 1 || b_dims[0] != out_features) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }

    // Create descriptor
    auto desc = new Descriptor();
    desc->device_type = INFINI_DEVICE_CPU;
    desc->_handle = reinterpret_cast<device::cpu::Handle *>(handle);

    // Save tensor descriptors
    desc->_x_desc = x_desc;
    desc->_w_desc = w_desc;
    if (b_desc) {
        desc->_b_desc = b_desc;
    } else {
        desc->_b_desc = nullptr;
    }
    desc->_y_desc = y_desc;
    
    // Save tensor shapes and data type to avoid accessing descriptors later
    desc->_x_dims = std::vector<int>(x_dims.begin(), x_dims.end());
    desc->_w_dims = std::vector<int>(w_dims.begin(), w_dims.end());
    desc->_dtype = x_dtype;

    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *w,
    const void *b,
    void *stream) const {

    // Use saved data type and shapes
    auto x_dtype = _dtype;
    const auto& x_dims = _x_dims;
    const auto& w_dims = _w_dims;
    
    // Call kernel based on data type
    if (x_dtype == INFINI_DTYPE_F32) {
        return linear_kernel<float>(
            static_cast<float *>(y),
            static_cast<const float *>(x),
            static_cast<const float *>(w),
            static_cast<const float *>(b),
            x_dims, w_dims);
    } else if (x_dtype == INFINI_DTYPE_F16) {
        return linear_kernel<fp16_t>(
            static_cast<fp16_t *>(y),
            static_cast<const fp16_t *>(x),
            static_cast<const fp16_t *>(w),
            static_cast<const fp16_t *>(b),
            x_dims, w_dims);
    } else if (x_dtype == INFINI_DTYPE_BF16) {
        return linear_kernel<bf16_t>(
            static_cast<bf16_t *>(y),
            static_cast<const bf16_t *>(x),
            static_cast<const bf16_t *>(w),
            static_cast<const bf16_t *>(b),
            x_dims, w_dims);
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::linear::cpu