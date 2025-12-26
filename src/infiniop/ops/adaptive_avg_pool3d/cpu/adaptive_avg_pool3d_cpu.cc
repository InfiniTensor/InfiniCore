#include "adaptive_avg_pool3d_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <array>
namespace op::adaptive_avg_pool3d::cpu {
Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    size_t *output_size) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto info = AdaptiveAvgPool3DInfo::create(y_desc, x_desc, output_size);
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
infiniStatus_t calculateAdaptiveAvgPool3D(
    const AdaptiveAvgPool3DInfo &info,
    Tdata *y,
    const Tdata *x) {
    std::array<size_t, 5> y_strides;
    y_strides[size_t(4)] = 1;
    y_strides[size_t(3)] = info.y_w * y_strides[size_t(4)];
    y_strides[size_t(2)] = info.y_h * y_strides[size_t(3)];
    y_strides[size_t(1)] = info.y_d * y_strides[size_t(2)];
    y_strides[size_t(0)] = info.C * y_strides[size_t(1)];
    {
#pragma omp for collapse(5)
        for (ptrdiff_t n = 0; n < ptrdiff_t(info.N); n++) {
            for (ptrdiff_t c = 0; c < ptrdiff_t(info.C); c++) {
                for (ptrdiff_t od = 0; od < ptrdiff_t(info.y_d); od++) {
                    for (ptrdiff_t oh = 0; oh < ptrdiff_t(info.y_h); oh++) {
                        for (ptrdiff_t ow = 0; ow < ptrdiff_t(info.y_w); ow++) {
                            size_t x_start_d = od * info.x_d / info.y_d;
                            size_t x_end_d = ((od + 1) * info.x_d + info.y_d - 1) / info.y_d;
                            size_t x_start_h = oh * info.x_h / info.y_h;
                            size_t x_end_h = ((oh + 1) * info.x_h + info.y_h - 1) / info.y_h;
                            size_t x_start_w = ow * info.x_w / info.y_w;
                            size_t x_end_w = ((ow + 1) * info.x_w + info.y_w - 1) / info.y_w;
                            size_t count = (x_end_d - x_start_d) * (x_end_h - x_start_h) * (x_end_w - x_start_w);

                            // Handle floating point types with casting
                            if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
                                float sum = 0.0f;
                                for (size_t id = x_start_d; id < x_end_d; id++) {
                                    for (size_t ih = x_start_h; ih < x_end_h; ih++) {
                                        for (size_t iw = x_start_w; iw < x_end_w; iw++) {
                                            size_t x_offset = n * info.x_strides[0] + c * info.x_strides[1] + id * info.x_strides[2] + ih * info.x_strides[3] + iw * info.x_strides[4];
                                            sum += utils::cast<float>(x[x_offset]);
                                        }
                                    }
                                }
                                size_t y_offset = n * y_strides[0] + c * y_strides[1] + od * y_strides[2] + oh * y_strides[3] + ow * y_strides[4];
                                y[y_offset] = utils::cast<Tdata>(sum / static_cast<float>(count));
                            } else {
                                Tdata sum = (Tdata)0;
                                for (size_t id = x_start_d; id < x_end_d; id++) {
                                    for (size_t ih = x_start_h; ih < x_end_h; ih++) {
                                        for (size_t iw = x_start_w; iw < x_end_w; iw++) {
                                            size_t x_offset = n * info.x_strides[0] + c * info.x_strides[1] + id * info.x_strides[2] + ih * info.x_strides[3] + iw * info.x_strides[4];
                                            sum += x[x_offset];
                                        }
                                    }
                                }
                                size_t y_offset = n * y_strides[0] + c * y_strides[1] + od * y_strides[2] + oh * y_strides[3] + ow * y_strides[4];
                                // For integer types, we might want to handle division differently
                                y[y_offset] = sum / static_cast<Tdata>(count);
                            }
                        }
                    }
                }
            }
        }
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return calculateAdaptiveAvgPool3D<fp16_t>(_info, reinterpret_cast<fp16_t *>(y), reinterpret_cast<const fp16_t *>(x));
    case INFINI_DTYPE_F32:
        return calculateAdaptiveAvgPool3D<float>(_info, reinterpret_cast<float *>(y), reinterpret_cast<const float *>(x));
    case INFINI_DTYPE_F64:
        return calculateAdaptiveAvgPool3D<double>(_info, reinterpret_cast<double *>(y), reinterpret_cast<const double *>(x));
    case INFINI_DTYPE_BF16:
        return calculateAdaptiveAvgPool3D<bf16_t>(_info, reinterpret_cast<bf16_t *>(y), reinterpret_cast<const bf16_t *>(x));
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::adaptive_avg_pool3d::cpu