#include "asum_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::asum::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t result_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto info = AsumInfo::createAsumInfo(x_desc, result_desc);
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
infiniStatus_t calculateAsum(
    const AsumInfo &info,
    const Tdata *x,
    Tdata *result) {

    const ptrdiff_t size = info.getSize();
    const ptrdiff_t incx = info.getIncx();

    if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
        float total_sum = 0.0;

        for (ptrdiff_t i = 0; i < size; ++i) {
            total_sum += std::abs(utils::cast<float>(x[i * incx]));
        }

        result[0] = utils::cast<Tdata>(total_sum);
    } else {
        Tdata total_sum = 0.0;

        for (ptrdiff_t i = 0; i < size; ++i) {
            total_sum += std::abs(x[i * incx]);
        }

        result[0] = total_sum;
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_ASUM(TDATA)       \
    calculateAsum(_info,            \
                  (const TDATA *)x, \
                  (TDATA *)result)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *x,
    void *result,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F16:
        return CALCULATE_ASUM(fp16_t);
    case INFINI_DTYPE_BF16:
        return CALCULATE_ASUM(bf16_t);
    case INFINI_DTYPE_F32:
        return CALCULATE_ASUM(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_ASUM(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_ASUM

} // namespace op::asum::cpu
