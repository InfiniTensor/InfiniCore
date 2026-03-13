#include "eye_cpu.h"
#include "../info.h"
#include "../../../../utils/result.hpp"

namespace op::eye::cpu {

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t y_desc) {
    (void)handle_;
    auto info_result = EyeInfo::create(y_desc);
    CHECK_RESULT(info_result);
    *desc_ptr = new Descriptor(info_result.take(), INFINI_DEVICE_CPU, 0);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
static void eye_fill(void *y, size_t n, size_t m) {
    auto *out = static_cast<T *>(y);
    T one_val = utils::cast<T>(1.0f);
    T zero_val = utils::cast<T>(0.0f);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            out[i * m + j] = (i == j) ? one_val : zero_val;
        }
    }
}

infiniStatus_t Descriptor::calculate(void *workspace,
                                    size_t workspace_size,
                                    void *y,
                                    void *stream) const {
    (void)workspace;
    (void)workspace_size;
    (void)stream;

    size_t n = _info.n;
    size_t m = _info.m;

    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        eye_fill<fp16_t>(y, n, m);
        break;
    case INFINI_DTYPE_F32:
        eye_fill<float>(y, n, m);
        break;
    case INFINI_DTYPE_F64:
        eye_fill<double>(y, n, m);
        break;
    case INFINI_DTYPE_BF16:
        eye_fill<bf16_t>(y, n, m);
        break;
    case INFINI_DTYPE_I32:
        eye_fill<int32_t>(y, n, m);
        break;
    case INFINI_DTYPE_I64:
        eye_fill<int64_t>(y, n, m);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::eye::cpu
