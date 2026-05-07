#include "axpy_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::axpy::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = AxpyInfo::createAxpyInfo(alpha_desc, x_desc, y_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateAxpy(
    const AxpyInfo &info,
    const Tdata *alpha,
    const Tdata *x,
    Tdata *y) {

    const size_t n = info.n;
    const ptrdiff_t incx = info.incx;
    const ptrdiff_t incy = info.incy;

    if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
        const float alpha_f = utils::cast<float>(alpha[0]);
        for (size_t i = 0; i < n; ++i) {
            const ptrdiff_t x_idx = utils::cast<ptrdiff_t>(i) * incx;
            const ptrdiff_t y_idx = utils::cast<ptrdiff_t>(i) * incy;
            const float x_f = utils::cast<float>(x[x_idx]);
            const float y_f = utils::cast<float>(y[y_idx]);
            y[y_idx] = utils::cast<Tdata>(alpha_f * x_f + y_f);
        }
    } else {
        const Tdata alpha_v = alpha[0];
        for (size_t i = 0; i < n; ++i) {
            const ptrdiff_t x_idx = utils::cast<ptrdiff_t>(i) * incx;
            const ptrdiff_t y_idx = utils::cast<ptrdiff_t>(i) * incy;
            y[y_idx] = alpha_v * x[x_idx] + y[y_idx];
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_AXPY(TDATA)           \
    calculateAxpy(_info,                \
                  (const TDATA *)alpha, \
                  (const TDATA *)x,     \
                  (TDATA *)y)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *x,
    void *y,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        return CALCULATE_AXPY(fp16_t);
    case INFINI_DTYPE_F32:
        return CALCULATE_AXPY(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_AXPY(double);
    case INFINI_DTYPE_BF16:
        return CALCULATE_AXPY(bf16_t);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_AXPY

} // namespace op::axpy::cpu
