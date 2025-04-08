#include "rope_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::rope::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t t_desc,
    infiniopTensorDescriptor_t pos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t cos_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto t_dtype = t_desc->dtype();

    CHECK_DTYPE(t_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
    CHECK_DTYPE(pos_desc->dtype(), INFINI_DTYPE_U8, INFINI_DTYPE_U16, INFINI_DTYPE_U32, INFINI_DTYPE_U64);

    RoPEInfo info = {};
    CHECK_STATUS(createRoPEInfo(info, t_desc, pos_desc, sin_desc, cos_desc));

    // Create descriptor
    *desc_ptr = new Descriptor(
        std::move(info),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

size_t read_pos(infiniDtype_t ty, uint8_t const *pos) {
    switch (ty) {
    case INFINI_DTYPE_U8:
        return *pos;
    case INFINI_DTYPE_U16:
        return *reinterpret_cast<const uint16_t *>(pos);
    case INFINI_DTYPE_U32:
        return *reinterpret_cast<const uint32_t *>(pos);
    case INFINI_DTYPE_U64:
        return *reinterpret_cast<const uint64_t *>(pos);
    default:
        // unreachable
        std::abort();
    }
}

template <class T>
void rope_ptr(uint8_t *t, uint8_t const *sin, uint8_t const *cos) {
    auto &a = reinterpret_cast<T *>(t)[0],
         &b = reinterpret_cast<T *>(t)[1];
    auto sin_ = *reinterpret_cast<const float *>(sin),
         cos_ = *reinterpret_cast<const float *>(cos);
    auto a_ = a, b_ = b;
    a = a_ * cos_ - b_ * sin_;
    b = a_ * sin_ + b_ * cos_;
}

template <>
void rope_ptr<fp16_t>(uint8_t *t, uint8_t const *sin, uint8_t const *cos) {
    auto &a = reinterpret_cast<fp16_t *>(t)[0],
         &b = reinterpret_cast<fp16_t *>(t)[1];
    auto sin_ = *reinterpret_cast<const float *>(sin),
         cos_ = *reinterpret_cast<const float *>(cos);
    auto a_ = utils::cast<float>(a),
         b_ = utils::cast<float>(b);
    a = utils::cast<fp16_t>(a_ * cos_ - b_ * sin_);
    b = utils::cast<fp16_t>(a_ * sin_ + b_ * cos_);
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *t,
    const void *pos_ids,
    const float *sin_table,
    const float *cos_table,
    void *stream) const {

    const auto t_ = reinterpret_cast<uint8_t *>(t);
    const auto pos_ = reinterpret_cast<const uint8_t *>(pos_ids);
    const auto sin_ = reinterpret_cast<const uint8_t *>(sin_table);
    const auto cos_ = reinterpret_cast<const uint8_t *>(cos_table);
    const auto unit_t = infiniSizeOf(_info.ty_t),
               unit_pos = infiniSizeOf(_info.ty_pos);
    constexpr size_t unit_sin_cos = sizeof(*sin_table);

    for (size_t i = 0; i < _info.nt; ++i) {
        auto const t__ = t_ + i * _info.s_nt * unit_t;
        auto const pos__ = read_pos(_info.ty_pos, pos_ + i * _info.s_np * unit_pos);
        auto const sin__ = sin_ + pos__ * _info.s_nsin * unit_sin_cos;
        auto const cos__ = cos_ + pos__ * _info.s_ncos * unit_sin_cos;
        if (pos__ >= _info.nsin || pos__ >= _info.ncos) {
            // sin cos 表容量不足
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        for (size_t j = 0; j < _info.nh; ++j) {
            for (size_t k = 0; k < _info.dh / 2; ++k) {
                const auto t___ = t__ + (j * _info.s_nh + k * 2) * unit_t;
                const auto sin___ = sin__ + k * _info.s_dsin * unit_sin_cos;
                const auto cos___ = cos__ + k * _info.s_dcos * unit_sin_cos;
                switch (_info.ty_t) {
                case INFINI_DTYPE_F16:
                    rope_ptr<fp16_t>(t___, sin___, cos___);
                    break;
                case INFINI_DTYPE_F32:
                    rope_ptr<float>(t___, sin___, cos___);
                    break;
                case INFINI_DTYPE_F64:
                    rope_ptr<double>(t___, sin___, cos___);
                    break;
                default:
                    // unreachable
                    std::abort();
                }
            }
        }
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::rope::cpu
