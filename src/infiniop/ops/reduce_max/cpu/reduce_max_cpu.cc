#include "reduce_max_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"

namespace op::reduce_max::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    size_t dim) {
    auto result = ReduceMaxInfo::create(y_desc, x_desc, dim);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t reduce_max(const ReduceMaxInfo *info, T *y, const T *x) {
    const size_t cols = info->shape[3]; // 规约维度
    const ptrdiff_t y_batch_stride = info->y_strides[0];
    const ptrdiff_t y_channel_stride = info->y_strides[1];
    const ptrdiff_t y_row_stride = info->y_strides[2];
    const ptrdiff_t x_batch_stride = info->x_strides[0];
    const ptrdiff_t x_channel_stride = info->x_strides[1];
    const ptrdiff_t x_row_stride = info->x_strides[2];
    const ptrdiff_t x_col_stride = info->x_strides[3];
#ifdef _WIN32
    const ptrdiff_t batch_size = static_cast<ptrdiff_t>(info->shape[0]);
    const ptrdiff_t channels = static_cast<ptrdiff_t>(info->shape[1]);
    const ptrdiff_t rows = static_cast<ptrdiff_t>(info->shape[2]);
#pragma omp parallel for
    for (ptrdiff_t batch = 0; batch < batch_size; ++batch) {
        for (ptrdiff_t channel = 0; channel < channels; ++channel) {
            for (ptrdiff_t row = 0; row < rows; ++row) {
                const T *input_start = x + batch * x_batch_stride
                                     + channel * x_channel_stride
                                     + row * x_row_stride;
                T *output_ptr = y + batch * y_batch_stride
                              + channel * y_channel_stride
                              + row * y_row_stride;

                float max = op::common_cpu::reduce_op::max(input_start, cols, x_col_stride);

                if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                    *output_ptr = utils::cast<T>(max);
                } else {
                    *output_ptr = max;
                }
            }
        }
    }
#else
    const size_t batch_size = info->shape[0];
    const size_t channels = info->shape[1];
    const size_t rows = info->shape[2];
#pragma omp parallel for collapse(3)
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t channel = 0; channel < channels; ++channel) {
            for (size_t row = 0; row < rows; ++row) {
                const T *input_start = x + batch * x_batch_stride
                                     + channel * x_channel_stride
                                     + row * x_row_stride;
                T *output_ptr = y + batch * y_batch_stride
                              + channel * y_channel_stride
                              + row * y_row_stride;

                float max = op::common_cpu::reduce_op::max(input_start, cols, x_col_stride);

                if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                    *output_ptr = utils::cast<T>(max);
                } else {
                    *output_ptr = max;
                }
            }
        }
    }
#endif //_WIN32
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    if (_info.dtype == INFINI_DTYPE_F16) {
        CHECK_STATUS(reduce_max<fp16_t>(&_info, (fp16_t *)y, (const fp16_t *)x));
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        CHECK_STATUS(reduce_max<bf16_t>(&_info, (bf16_t *)y, (const bf16_t *)x));
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CHECK_STATUS(reduce_max<float>(&_info, (float *)y, (const float *)x));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::reduce_max::cpu
