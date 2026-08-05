#include "mxfp4_dequantize_cpu.h"

#include "../../../../utils/custom_types.h"
#include "../../../devices/cpu/cpu_handle.h"

#include <cmath>
#include <cstdint>

namespace op::mxfp4_dequantize::cpu {
namespace {

float decode_e2m1(uint8_t value) {
    static constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float magnitude = magnitudes[value & 0x7];
    return value & 0x8 ? -magnitude : magnitude;
}

template <typename T>
void dequantize(T *out,
                const uint8_t *packed,
                const uint8_t *scales,
                const Mxfp4DequantizeInfo &info) {
    const size_t packed_width = info.logical_width / 2;
    const size_t scales_width = info.logical_width / 32;
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(info.rows); ++row) {
        for (size_t packed_col = 0; packed_col < packed_width; ++packed_col) {
            const uint8_t byte = packed[row * packed_width + packed_col];
            const int exponent = static_cast<int>(scales[row * scales_width + packed_col / 16]) - 127;
            const size_t out_col = packed_col * 2;
            out[row * info.logical_width + out_col] = utils::cast<T>(std::ldexp(decode_e2m1(byte & 0xf), exponent));
            out[row * info.logical_width + out_col + 1] = utils::cast<T>(std::ldexp(decode_e2m1(byte >> 4), exponent));
        }
    }
}

} // namespace

struct Descriptor::Opaque {};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t packed_desc,
    infiniopTensorDescriptor_t scales_desc) {
    auto info = Mxfp4DequantizeInfo::create(out_desc, packed_desc, scales_desc);
    CHECK_RESULT(info);
    *desc_ptr = new Descriptor(new Opaque{}, info.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *, size_t, void *out,
    const void *packed, const void *scales, void *) const {
    switch (_info.output_dtype) {
    case INFINI_DTYPE_F16:
        dequantize(reinterpret_cast<fp16_t *>(out),
                   reinterpret_cast<const uint8_t *>(packed),
                   reinterpret_cast<const uint8_t *>(scales), _info);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        dequantize(reinterpret_cast<bf16_t *>(out),
                   reinterpret_cast<const uint8_t *>(packed),
                   reinterpret_cast<const uint8_t *>(scales), _info);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        dequantize(reinterpret_cast<float *>(out),
                   reinterpret_cast<const uint8_t *>(packed),
                   reinterpret_cast<const uint8_t *>(scales), _info);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::mxfp4_dequantize::cpu
