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
    auto info = SwapInfo::createSwapInfo(x_desc, y_desc);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        info.take(),
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

    const ptrdiff_t size = info.getSize();
    const ptrdiff_t incx = info.getIncx();
    const ptrdiff_t incy = info.getIncy();

#pragma omp parallel for if (size > 1024)
    for (ptrdiff_t i = 0; i < size; ++i) {
        const ptrdiff_t x_idx = i * incx;
        const ptrdiff_t y_idx = i * incy;
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

    switch (_info.getDtype()) {
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
