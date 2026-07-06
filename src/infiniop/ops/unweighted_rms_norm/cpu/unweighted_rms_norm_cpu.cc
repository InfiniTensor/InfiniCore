#include "unweighted_rms_norm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"

namespace op::unweighted_rms_norm::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    float epsilon) {
    auto result = UnweightedRMSNormInfo::create(y_desc, x_desc, epsilon);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t unweightedRMSNormFloat(const UnweightedRMSNormInfo *info, T *y, const T *x) {
    const size_t outer_size = info->outer_size;
    const size_t dim = info->last_dim;

#pragma omp parallel for
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(outer_size); ++row) {
        const T *x_ptr = x + static_cast<size_t>(row) * dim;
        T *y_ptr = y + static_cast<size_t>(row) * dim;
        T ss = op::common_cpu::reduce_op::sumSquared(x_ptr, dim, 1);
        T rms = (T)1 / std::sqrt(ss / (T)(dim) + (T)(info->epsilon));
        for (size_t i = 0; i < dim; ++i) {
            y_ptr[i] = x_ptr[i] * rms;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t unweightedRMSNormHalf(const UnweightedRMSNormInfo *info, T *y, const T *x) {
    static_assert(std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value,
                  "T must be fp16_t or bf16_t");

    const size_t outer_size = info->outer_size;
    const size_t dim = info->last_dim;

#pragma omp parallel for
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(outer_size); ++row) {
        const T *x_ptr = x + static_cast<size_t>(row) * dim;
        T *y_ptr = y + static_cast<size_t>(row) * dim;
        float ss = op::common_cpu::reduce_op::sumSquared(x_ptr, dim, 1);
        float rms = 1.f / std::sqrt(ss / static_cast<float>(dim) + info->epsilon);
        for (size_t i = 0; i < dim; ++i) {
            y_ptr[i] = utils::cast<T>(utils::cast<float>(x_ptr[i]) * rms);
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x,
    void *stream) const {
    if (_info.atype == INFINI_DTYPE_F16) {
        CHECK_STATUS(unweightedRMSNormHalf(&_info, (fp16_t *)y, (const fp16_t *)x));
    } else if (_info.atype == INFINI_DTYPE_BF16) {
        CHECK_STATUS(unweightedRMSNormHalf(&_info, (bf16_t *)y, (const bf16_t *)x));
    } else if (_info.atype == INFINI_DTYPE_F32) {
        CHECK_STATUS(unweightedRMSNormFloat(&_info, (float *)y, (const float *)x));
    } else if (_info.atype == INFINI_DTYPE_F64) {
        CHECK_STATUS(unweightedRMSNormFloat(&_info, (double *)y, (const double *)x));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::unweighted_rms_norm::cpu
