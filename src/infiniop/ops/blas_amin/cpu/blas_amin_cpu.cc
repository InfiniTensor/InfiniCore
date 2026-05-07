#include "blas_amin_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::blas_amin::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t result_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = BlasAminInfo::createBlasAminInfo(x_desc, result_desc);
    CHECK_RESULT(result);

    // Create descriptor
    *desc_ptr = new Descriptor(
        result.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateBlasAmin(
    const BlasAminInfo &info,
    const Tdata *x,
    int *result) {

    const size_t n = info.n;
    const ptrdiff_t incx = info.incx;

    if (n < 1 || incx == 0) {
        result[0] = 0;
        return INFINI_STATUS_SUCCESS;
    }

    size_t min_index = 0;
    if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
        float min_value = std::abs(utils::cast<float>(x[0]));

        for (size_t i = 1; i < n; ++i) {
            const ptrdiff_t idx = utils::cast<ptrdiff_t>(i) * incx;
            float current_value = std::abs(utils::cast<float>(x[idx]));
            if (current_value < min_value) {
                min_value = current_value;
                min_index = i;
            }
        }
    } else {
        Tdata min_value = std::abs(x[0]);

        for (size_t i = 1; i < n; ++i) {
            const ptrdiff_t idx = utils::cast<ptrdiff_t>(i) * incx;
            Tdata current_value = std::abs(x[idx]);
            if (current_value < min_value) {
                min_value = current_value;
                min_index = i;
            }
        }
    }

    result[0] = utils::cast<int>(min_index) + 1;

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_BLAS_AMIN(TDATA)      \
    calculateBlasAmin(_info,            \
                      (const TDATA *)x, \
                      (int *)result)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *x,
    void *result,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        return CALCULATE_BLAS_AMIN(fp16_t);
    case INFINI_DTYPE_F32:
        return CALCULATE_BLAS_AMIN(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_BLAS_AMIN(double);
    case INFINI_DTYPE_BF16:
        return CALCULATE_BLAS_AMIN(bf16_t);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_BLAS_AMIN

} // namespace op::blas_amin::cpu
