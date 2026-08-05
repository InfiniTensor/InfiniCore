#include "fused_moe_mxfp4_cpu.h"

#include "../../../../utils/custom_types.h"
#include "../../../devices/cpu/cpu_handle.h"

#include <cmath>
#include <cstdint>

namespace op::fused_moe_mxfp4::cpu {
namespace {

float decode_e2m1(uint8_t code) {
    static constexpr float magnitudes[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float magnitude = magnitudes[code & 0x7];
    return code & 0x8 ? -magnitude : magnitude;
}

template <typename T>
float packed_dot(const T *input,
                 const uint8_t *packed,
                 const uint8_t *scales,
                 size_t K) {
    float sum = 0.0f;
    for (size_t packed_k = 0; packed_k < K / 2; ++packed_k) {
        const uint8_t byte = packed[packed_k];
        const int exponent = static_cast<int>(scales[packed_k / 16]) - 127;
        const size_t k = packed_k * 2;
        sum += utils::cast<float>(input[k])
             * std::ldexp(decode_e2m1(byte & 0xf), exponent);
        sum += utils::cast<float>(input[k + 1])
             * std::ldexp(decode_e2m1(byte >> 4), exponent);
    }
    return sum;
}

float activate(float gate, float up, infiniopFusedMoeActivation_t activation) {
    if (activation == INFINIOP_FUSED_MOE_ACT_SITUGLU) {
        constexpr float beta = 4.0f;
        constexpr float linear_beta = 25.0f;
        const float situ_gate = beta * std::tanh(gate / beta) / (1.0f + std::exp(-gate));
        const float bounded_up = linear_beta * std::tanh(up / linear_beta);
        return situ_gate * bounded_up;
    }
    return gate / (1.0f + std::exp(-gate)) * up;
}

template <typename T>
void fused_moe(T *output,
               T *activated,
               const T *input,
               const int32_t *selected_experts,
               const float *routing_weights,
               const uint8_t *w13_packed,
               const uint8_t *w13_scale,
               const uint8_t *w2_packed,
               const uint8_t *w2_scale,
               const FusedMoeMxfp4Info &info) {
    const size_t w13_packed_row = info.hidden_size / 2;
    const size_t w13_scale_row = info.hidden_size / 32;
    const size_t w2_packed_row = info.intermediate_size / 2;
    const size_t w2_scale_row = info.intermediate_size / 32;

    const ptrdiff_t activated_count
        = static_cast<ptrdiff_t>(info.routeCount() * info.intermediate_size);
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
    for (ptrdiff_t flat_index = 0; flat_index < activated_count; ++flat_index) {
        const size_t route = static_cast<size_t>(flat_index) / info.intermediate_size;
        const size_t i = static_cast<size_t>(flat_index) % info.intermediate_size;
        const int32_t expert = selected_experts[route];
        if (expert < 0 || static_cast<size_t>(expert) >= info.num_experts) {
            activated[route * info.intermediate_size + i] = utils::cast<T>(0.0f);
            continue;
        }
        const size_t token = route / info.topk;
        const size_t gate_row = (static_cast<size_t>(expert) * 2 * info.intermediate_size + i);
        const size_t up_row = gate_row + info.intermediate_size;
        const float gate = packed_dot(
            input + token * info.hidden_size,
            w13_packed + gate_row * w13_packed_row,
            w13_scale + gate_row * w13_scale_row,
            info.hidden_size);
        const float up = packed_dot(
            input + token * info.hidden_size,
            w13_packed + up_row * w13_packed_row,
            w13_scale + up_row * w13_scale_row,
            info.hidden_size);
        activated[route * info.intermediate_size + i]
            = utils::cast<T>(activate(gate, up, info.activation));
    }

    const ptrdiff_t output_count
        = static_cast<ptrdiff_t>(info.num_tokens * info.hidden_size);
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
    for (ptrdiff_t flat_index = 0; flat_index < output_count; ++flat_index) {
        const size_t token = static_cast<size_t>(flat_index) / info.hidden_size;
        const size_t h = static_cast<size_t>(flat_index) % info.hidden_size;
        float sum = 0.0f;
        for (size_t route_index = 0; route_index < info.topk; ++route_index) {
            const size_t route = token * info.topk + route_index;
            const int32_t expert = selected_experts[route];
            if (expert < 0 || static_cast<size_t>(expert) >= info.num_experts) {
                continue;
            }
            const size_t weight_row = (static_cast<size_t>(expert) * info.hidden_size + h);
            sum += routing_weights[route]
                 * packed_dot(
                       activated + route * info.intermediate_size,
                       w2_packed + weight_row * w2_packed_row,
                       w2_scale + weight_row * w2_scale_row,
                       info.intermediate_size);
        }
        output[token * info.hidden_size + h] = utils::cast<T>(sum);
    }
    return;
}

size_t dtype_size(infiniDtype_t dtype) {
    return dtype == INFINI_DTYPE_F32 ? sizeof(float) : sizeof(uint16_t);
}

} // namespace

struct Descriptor::Opaque {};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t selected_experts_desc,
    infiniopTensorDescriptor_t routing_weights_desc,
    infiniopTensorDescriptor_t w13_packed_desc,
    infiniopTensorDescriptor_t w13_scale_desc,
    infiniopTensorDescriptor_t w2_packed_desc,
    infiniopTensorDescriptor_t w2_scale_desc,
    infiniopFusedMoeActivation_t activation) {
    auto info = FusedMoeMxfp4Info::create(
        output_desc, input_desc, selected_experts_desc, routing_weights_desc,
        w13_packed_desc, w13_scale_desc, w2_packed_desc, w2_scale_desc, activation);
    CHECK_RESULT(info);
    auto value = info.take();
    const size_t workspace_size = value.routeCount() * value.intermediate_size * dtype_size(value.dtype);
    *desc_ptr = new Descriptor(
        new Opaque{}, value, workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *selected_experts,
    const void *routing_weights,
    const void *w13_packed,
    const void *w13_scale,
    const void *w2_packed,
    const void *w2_scale,
    void *) const {
    CHECK_OR_RETURN(workspace_size >= _workspace_size && workspace != nullptr,
                    INFINI_STATUS_INSUFFICIENT_WORKSPACE);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        fused_moe(reinterpret_cast<fp16_t *>(output),
                  reinterpret_cast<fp16_t *>(workspace),
                  reinterpret_cast<const fp16_t *>(input),
                  reinterpret_cast<const int32_t *>(selected_experts),
                  reinterpret_cast<const float *>(routing_weights),
                  reinterpret_cast<const uint8_t *>(w13_packed),
                  reinterpret_cast<const uint8_t *>(w13_scale),
                  reinterpret_cast<const uint8_t *>(w2_packed),
                  reinterpret_cast<const uint8_t *>(w2_scale), _info);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        fused_moe(reinterpret_cast<bf16_t *>(output),
                  reinterpret_cast<bf16_t *>(workspace),
                  reinterpret_cast<const bf16_t *>(input),
                  reinterpret_cast<const int32_t *>(selected_experts),
                  reinterpret_cast<const float *>(routing_weights),
                  reinterpret_cast<const uint8_t *>(w13_packed),
                  reinterpret_cast<const uint8_t *>(w13_scale),
                  reinterpret_cast<const uint8_t *>(w2_packed),
                  reinterpret_cast<const uint8_t *>(w2_scale), _info);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        fused_moe(reinterpret_cast<float *>(output),
                  reinterpret_cast<float *>(workspace),
                  reinterpret_cast<const float *>(input),
                  reinterpret_cast<const int32_t *>(selected_experts),
                  reinterpret_cast<const float *>(routing_weights),
                  reinterpret_cast<const uint8_t *>(w13_packed),
                  reinterpret_cast<const uint8_t *>(w13_scale),
                  reinterpret_cast<const uint8_t *>(w2_packed),
                  reinterpret_cast<const uint8_t *>(w2_scale), _info);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::fused_moe_mxfp4::cpu
