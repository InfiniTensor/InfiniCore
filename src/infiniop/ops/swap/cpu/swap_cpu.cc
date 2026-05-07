#include "swap_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::swap::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = SwapInfo::createSwapInfo(x_desc, y_desc);
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
infiniStatus_t calculateSwap(
    const SwapInfo &info,
    Tdata *x,
    Tdata *y) {

    const size_t n = info.n;
    const ptrdiff_t incx = info.incx;
    const ptrdiff_t incy = info.incy;

    for (size_t i = 0; i < n; ++i) {
        const ptrdiff_t x_idx = utils::cast<ptrdiff_t>(i) * incx;
        const ptrdiff_t y_idx = utils::cast<ptrdiff_t>(i) * incy;
        Tdata temp = x[x_idx];
        x[x_idx] = y[y_idx];
        y[y_idx] = temp;
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_SWAP(TDATA) \
    calculateSwap(_info,      \
                  (TDATA *)x, \
                  (TDATA *)y)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *x,
    void *y,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        return CALCULATE_SWAP(fp16_t);
    case INFINI_DTYPE_BF16:
        return CALCULATE_SWAP(bf16_t);
    case INFINI_DTYPE_F32:
        return CALCULATE_SWAP(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_SWAP(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_SWAP

} // namespace op::swap::cpu
