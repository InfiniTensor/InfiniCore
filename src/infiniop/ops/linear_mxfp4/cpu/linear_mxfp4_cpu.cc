#include "linear_mxfp4_cpu.h"

#include "../../../../utils/custom_types.h"
#include "../../../devices/cpu/cpu_handle.h"

#include <cmath>
#include <cstdint>

namespace op::linear_mxfp4::cpu {
namespace {

float decode_e2m1(uint8_t code) {
    static constexpr float magnitudes[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float magnitude = magnitudes[code & 0x7];
    return code & 0x8 ? -magnitude : magnitude;
}

template <typename T>
void linear(T *output,
            const T *input,
            const uint8_t *packed_weight,
            const uint8_t *weight_scale,
            const T *bias,
            const LinearMxfp4Info &info) {
    const size_t packed_width = info.K / 2;
    const size_t scale_width = info.K / 32;
    const ptrdiff_t output_count = static_cast<ptrdiff_t>(info.M * info.N);
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
    for (ptrdiff_t flat_index = 0; flat_index < output_count; ++flat_index) {
        const size_t m = static_cast<size_t>(flat_index) / info.N;
        const size_t n = static_cast<size_t>(flat_index) % info.N;
        float sum = 0.0f;
        const auto *packed_row = packed_weight + n * packed_width;
        const auto *scale_row = weight_scale + n * scale_width;
        for (size_t packed_k = 0; packed_k < packed_width; ++packed_k) {
            const uint8_t packed = packed_row[packed_k];
            const int exponent = static_cast<int>(scale_row[packed_k / 16]) - 127;
            const size_t k = packed_k * 2;
            sum += utils::cast<float>(input[m * info.K + k])
                 * std::ldexp(decode_e2m1(packed & 0xf), exponent);
            sum += utils::cast<float>(input[m * info.K + k + 1])
                 * std::ldexp(decode_e2m1(packed >> 4), exponent);
        }
        sum *= info.alpha;
        if (bias != nullptr) {
            sum += utils::cast<float>(bias[n]);
        }
        output[m * info.N + n] = utils::cast<T>(sum);
    }
    return;
}

} // namespace

struct Descriptor::Opaque {};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t packed_weight_desc,
    infiniopTensorDescriptor_t weight_scale_desc,
    infiniopTensorDescriptor_t bias_desc,
    float alpha) {
    auto info = LinearMxfp4Info::create(
        output_desc, input_desc, packed_weight_desc, weight_scale_desc, bias_desc, alpha);
    CHECK_RESULT(info);
    *desc_ptr = new Descriptor(new Opaque{}, info.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *, size_t,
    void *output,
    const void *input,
    const void *packed_weight,
    const void *weight_scale,
    const void *bias,
    void *) const {
    const auto *packed_ptr = reinterpret_cast<const uint8_t *>(packed_weight);
    const auto *scale_ptr = reinterpret_cast<const uint8_t *>(weight_scale);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        linear(reinterpret_cast<fp16_t *>(output),
               reinterpret_cast<const fp16_t *>(input),
               packed_ptr, scale_ptr, reinterpret_cast<const fp16_t *>(bias), _info);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        linear(reinterpret_cast<bf16_t *>(output),
               reinterpret_cast<const bf16_t *>(input),
               packed_ptr, scale_ptr, reinterpret_cast<const bf16_t *>(bias), _info);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        linear(reinterpret_cast<float *>(output),
               reinterpret_cast<const float *>(input),
               packed_ptr, scale_ptr, reinterpret_cast<const float *>(bias), _info);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::linear_mxfp4::cpu
