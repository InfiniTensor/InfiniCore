#include "add_rms_norm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"

namespace op::add_rms_norm::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t x2_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {
    auto result = AddRMSNormInfo::create(y_desc, x1_desc, x2_desc, w_desc, epsilon);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t add_rmsnorm(const AddRMSNormInfo *info, T *y, const T *x1, const T *x2, const T *w) {
    const size_t batch_size = info->shape[0];
    const size_t nhead = info->shape.size() > 2 ? info->shape[1] : 1;
    const size_t dim = info->shape.back();
    const ptrdiff_t total_blocks = static_cast<ptrdiff_t>(batch_size * nhead);

#pragma omp parallel for
    for (ptrdiff_t block_idx = 0; block_idx < total_blocks; ++block_idx) {
        const size_t i = block_idx / nhead; // batch index
        const size_t j = block_idx % nhead; // head index

        const T *x1_ptr = x1 + i * info->x1_strides[0] + j * info->x1_strides[1];
        const T *x2_ptr = x2 + i * info->x2_strides[0] + j * info->x2_strides[1];
        T *y_ptr = y + i * info->y_strides[0] + j * info->y_strides[1];

        // [Reduce] sum of x^2 on last dimension
        T ss = op::common_cpu::reduce_op::sumBinomialSquare(x1_ptr, x2_ptr, dim, info->x1_strides.back(), info->x2_strides.back());

        // 1 / (sqrt(sum/dim + eps))
        T rms = (T)1 / std::sqrt(ss / (T)(dim) + (T)(info->epsilon));

        for (size_t k = 0; k < dim; k++) {
            y_ptr[k] = x_ptr[k] * w[k] * rms;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

template <typename T, typename Tw>
infiniStatus_t add_rmsnorm_half(const AddRMSNormInfo *info, T *y, const T *x1, const T *x2, const Tw *w) {
    static_assert(std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value,
                  "T must be fp16_t or bf16_t");

    const size_t batch_size = info->shape[0];
    const size_t nhead = info->shape.size() > 2 ? info->shape[1] : 1;
    const size_t dim = info->shape.back();
    const ptrdiff_t total_blocks = static_cast<ptrdiff_t>(batch_size * nhead);

#pragma omp parallel for
    for (ptrdiff_t block_idx = 0; block_idx < total_blocks; ++block_idx) {
        const size_t i = block_idx / nhead; // batch index
        const size_t j = block_idx % nhead; // head index

        const T *x1_ptr = x1 + i * info->x1_strides[0] + j * info->x1_strides[1];
        const T *x2_ptr = x2 + i * info->x2_strides[0] + j * info->x2_strides[1];
        T *y_ptr = y + i * info->y_strides[0] + j * info->y_strides[1];

        // [Reduce] sum of x^2 on last dimension
        float ss = op::common_cpu::reduce_op::sumBinomialSquare(x1_ptr, x2_ptr, dim, info->x1_strides.back(), info->x2_strides.back());

        // 1 / (sqrt(sum/dim + eps))
        float rms = 1.f / std::sqrt(ss / (float)(dim) + info->epsilon);

        for (size_t k = 0; k < dim; k++) {
            if constexpr (std::is_same<Tw, float>::value) {
                float val = (utils::cast<float>(x1_ptr[k]) + utils::cast<float>(x2_ptr[k])) * w[k] * rms;
                y_ptr[k] = utils::cast<T>(val);
            } else if constexpr (std::is_same<Tw, T>::value || std::is_same_v<Tw, fp16_t> || std::is_same_v<Tw, bf16_t>) {
                float val = (utils::cast<float>(x1_ptr[k]) + utils::cast<float>(x2_ptr[k])) * utils::cast<float>(w[k]) * rms;
                y_ptr[k] = utils::cast<T>(val);
            } else {
                std::abort();
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x1, const void *x2, const void *w,
    void *stream) const {
    if (_info.atype == INFINI_DTYPE_F16) {
        if (_info.wtype == INFINI_DTYPE_F16) {
            CHECK_STATUS(add_rmsnorm_half(&_info, (fp16_t *)y, (const fp16_t *)x1, (const fp16_t *)x2, (const fp16_t *)w));
        } else if (_info.wtype == INFINI_DTYPE_F32) {
            CHECK_STATUS(add_rmsnorm_half(&_info, (fp16_t *)y, (const fp16_t *)x1, (const fp16_t *)x2, (const float *)w));
        } else if (_info.wtype == INFINI_DTYPE_BF16) {
            CHECK_STATUS(add_rmsnorm_half(&_info, (fp16_t *)y, (const fp16_t *)x1, (const fp16_t *)x2, (const bf16_t *)w));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (_info.atype == INFINI_DTYPE_BF16) {
        if (_info.wtype == INFINI_DTYPE_BF16) {
            CHECK_STATUS(add_rmsnorm_half(&_info, (bf16_t *)y, (const bf16_t *)x1, (const bf16_t *)x2, (const bf16_t *)w));
        } else if (_info.wtype == INFINI_DTYPE_F32) {
            CHECK_STATUS(add_rmsnorm_half(&_info, (bf16_t *)y, (const bf16_t *)x1, (const bf16_t *)x2, (const float *)w));
        } else if (_info.wtype == INFINI_DTYPE_F16) {
            CHECK_STATUS(add_rmsnorm_half(&_info, (bf16_t *)y, (const bf16_t *)x1, (const bf16_t *)x2, (const fp16_t *)w));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (_info.atype == INFINI_DTYPE_F32) {
        CHECK_STATUS(add_rmsnorm(&_info, (float *)y, (const float *)x1, (const float *)x2, (const float *)w));
    } else if (_info.atype == INFINI_DTYPE_F64) {
        CHECK_STATUS(add_rmsnorm(&_info, (double *)y, (const double *)x1, (const double *)x2, (const double *)w));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::add_rms_norm::cpu
