#include "hinge_embedding_loss_cpu.h"
#include "../../../utils.h"
#include <algorithm>
#include <cmath>

namespace op::hinge_embedding_loss::cpu {

utils::Result<HingeEmbeddingLossInfo> HingeEmbeddingLossInfo::create(
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    infiniopTensorDescriptor_t y_desc,
    double margin,
    int reduction) {

    auto input_shape = input_desc->shape();
    auto target_shape = target_desc->shape();
    auto y_shape = y_desc->shape();

    if (input_shape != target_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    Reduction red = static_cast<Reduction>(reduction);
    std::vector<size_t> expected_y_shape;
    if (red == Reduction::NONE) {
        expected_y_shape = input_shape;
    } else {
        // Mean or Sum: output is scalar
        expected_y_shape = {};
    }

    if (y_shape != expected_y_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    HingeEmbeddingLossInfo info;
    info.input_size = input_desc->numel();
    info.margin = margin;
    info.reduction = red;

    return utils::Result<HingeEmbeddingLossInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    double margin,
    int reduction) {

    auto dtype = input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto info_result = HingeEmbeddingLossInfo::create(input_desc, target_desc, y_desc, margin, reduction);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void hinge_embedding_loss_impl(
    const HingeEmbeddingLossInfo &info,
    T *y,
    const T *input,
    const T *target) {

    size_t n = info.input_size;
    T margin_val = utils::cast<T>(info.margin);

    if (info.reduction == Reduction::NONE) {
        // Element-wise loss
        for (size_t i = 0; i < n; ++i) {
            T t = target[i];
            T in = input[i];
            if (t > 0) {
                // target == 1: loss = max(0, margin - input)
                y[i] = std::max(utils::cast<T>(0.0), margin_val - in);
            } else {
                // target == -1: loss = max(0, input)
                y[i] = std::max(utils::cast<T>(0.0), in);
            }
        }
    } else {
        // Sum or Mean
        T sum = utils::cast<T>(0.0);
        for (size_t i = 0; i < n; ++i) {
            T t = target[i];
            T in = input[i];
            if (t > 0) {
                sum += std::max(utils::cast<T>(0.0), margin_val - in);
            } else {
                sum += std::max(utils::cast<T>(0.0), in);
            }
        }
        if (info.reduction == Reduction::MEAN) {
            y[0] = sum / utils::cast<T>(static_cast<double>(n));
        } else {
            y[0] = sum;
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *input,
    const void *target,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        hinge_embedding_loss_impl<fp16_t>(_info, reinterpret_cast<fp16_t *>(y),
                                         reinterpret_cast<const fp16_t *>(input),
                                         reinterpret_cast<const fp16_t *>(target));
        break;
    case INFINI_DTYPE_BF16:
        hinge_embedding_loss_impl<bf16_t>(_info, reinterpret_cast<bf16_t *>(y),
                                         reinterpret_cast<const bf16_t *>(input),
                                         reinterpret_cast<const bf16_t *>(target));
        break;
    case INFINI_DTYPE_F32:
        hinge_embedding_loss_impl<float>(_info, reinterpret_cast<float *>(y),
                                        reinterpret_cast<const float *>(input),
                                        reinterpret_cast<const float *>(target));
        break;
    case INFINI_DTYPE_F64:
        hinge_embedding_loss_impl<double>(_info, reinterpret_cast<double *>(y),
                                         reinterpret_cast<const double *>(input),
                                         reinterpret_cast<const double *>(target));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::hinge_embedding_loss::cpu
