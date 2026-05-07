#include "scal_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::scal::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = ScalInfo::createScalInfo(alpha_desc, x_desc);
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
infiniStatus_t calculateScal(
    const ScalInfo &info,
    const Tdata *alpha,
    Tdata *x) {

    const size_t n = info.n;
    const ptrdiff_t incx = info.incx;

    for (size_t i = 0; i < n; ++i) {
        const ptrdiff_t idx = utils::cast<ptrdiff_t>(i) * incx;

        if constexpr (std::is_same_v<Tdata, fp16_t> || std::is_same_v<Tdata, bf16_t>) {
            x[idx] = utils::cast<Tdata>(utils::cast<float>(x[idx]) * utils::cast<float>(alpha[0]));
        } else {
            x[idx] = x[idx] * alpha[0];
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_SCAL(TDATA)           \
    calculateScal(_info,                \
                  (const TDATA *)alpha, \
                  (TDATA *)x)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    void *x,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        return CALCULATE_SCAL(fp16_t);
    case INFINI_DTYPE_F32:
        return CALCULATE_SCAL(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_SCAL(double);
    case INFINI_DTYPE_BF16:
        return CALCULATE_SCAL(bf16_t);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_SCAL

} // namespace op::scal::cpu
