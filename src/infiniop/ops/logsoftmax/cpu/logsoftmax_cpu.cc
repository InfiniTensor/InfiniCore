#include "logsoftmax_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include <algorithm>
#include <cmath>

namespace op::logsoftmax::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {
    auto result = LogSoftmaxInfo::create(y_desc, x_desc);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tx, typename Ty>
infiniStatus_t logsoftmax(const LogSoftmaxInfo *info, Ty *y, const Tx *x) {
#pragma omp parallel for
    for (ptrdiff_t batch = 0; batch < ptrdiff_t(info->batch_size); batch++) {
        ptrdiff_t y_offset, x_offset;

        if (info->ndim == 3) {
            // For 3D tensors, convert linear batch index back to 2D indices
            ptrdiff_t batch_idx = batch / info->seq_len;
            ptrdiff_t seq_idx = batch % info->seq_len;
            y_offset = batch_idx * info->y_stride_0 + seq_idx * info->y_stride_1;
            x_offset = batch_idx * info->x_stride_0 + seq_idx * info->x_stride_1;
        } else {
            // For 2D tensors, use the flattened strides
            y_offset = batch * info->y_stride_b;
            x_offset = batch * info->x_stride_b;
        }

        Ty *y_ = y + y_offset;
        const Tx *x_ = x + x_offset;

        // Find max value for numerical stability
        float max_val;
        if constexpr (std::is_same<Tx, fp16_t>::value || std::is_same<Tx, bf16_t>::value) {
            max_val = op::common_cpu::reduce_op::max(x_, info->probs_size, info->x_stride_p);
        } else {
            max_val = op::common_cpu::reduce_op::max(x_, info->probs_size, info->x_stride_p);
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (size_t i = 0; i < info->probs_size; i++) {
            float x_val;
            if constexpr (std::is_same<Tx, fp16_t>::value || std::is_same<Tx, bf16_t>::value) {
                x_val = utils::cast<float>(x_[i * info->x_stride_p]);
            } else {
                x_val = x_[i * info->x_stride_p];
            }
            sum += std::exp(x_val - max_val);
        }

        // Compute log(sum)
        float log_sum = std::log(sum);

        // Compute log_softmax = x - max - log(sum)
        for (size_t i = 0; i < info->probs_size; i++) {
            float x_val;
            if constexpr (std::is_same<Tx, fp16_t>::value || std::is_same<Tx, bf16_t>::value) {
                x_val = utils::cast<float>(x_[i * info->x_stride_p]);
            } else {
                x_val = x_[i * info->x_stride_p];
            }

            float result = x_val - max_val - log_sum;

            if constexpr (std::is_same<Ty, fp16_t>::value || std::is_same<Ty, bf16_t>::value) {
                y_[i * info->y_stride_p] = utils::cast<Ty>(result);
            } else {
                y_[i * info->y_stride_p] = result;
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    // Handle different input/output dtype combinations
    if (_info.x_dtype == INFINI_DTYPE_F16) {
        if (_info.y_dtype == INFINI_DTYPE_F16) {
            return logsoftmax<fp16_t, fp16_t>(&_info, (fp16_t *)y, (const fp16_t *)x);
        } else if (_info.y_dtype == INFINI_DTYPE_BF16) {
            return logsoftmax<fp16_t, bf16_t>(&_info, (bf16_t *)y, (const fp16_t *)x);
        } else if (_info.y_dtype == INFINI_DTYPE_F32) {
            return logsoftmax<fp16_t, float>(&_info, (float *)y, (const fp16_t *)x);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (_info.x_dtype == INFINI_DTYPE_BF16) {
        if (_info.y_dtype == INFINI_DTYPE_F16) {
            return logsoftmax<bf16_t, fp16_t>(&_info, (fp16_t *)y, (const bf16_t *)x);
        } else if (_info.y_dtype == INFINI_DTYPE_BF16) {
            return logsoftmax<bf16_t, bf16_t>(&_info, (bf16_t *)y, (const bf16_t *)x);
        } else if (_info.y_dtype == INFINI_DTYPE_F32) {
            return logsoftmax<bf16_t, float>(&_info, (float *)y, (const bf16_t *)x);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (_info.x_dtype == INFINI_DTYPE_F32) {
        if (_info.y_dtype == INFINI_DTYPE_F16) {
            return logsoftmax<float, fp16_t>(&_info, (fp16_t *)y, (const float *)x);
        } else if (_info.y_dtype == INFINI_DTYPE_BF16) {
            return logsoftmax<float, bf16_t>(&_info, (bf16_t *)y, (const float *)x);
        } else if (_info.y_dtype == INFINI_DTYPE_F32) {
            return logsoftmax<float, float>(&_info, (float *)y, (const float *)x);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::logsoftmax::cpu
