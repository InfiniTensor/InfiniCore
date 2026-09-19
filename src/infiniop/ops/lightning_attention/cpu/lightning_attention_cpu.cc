#include "lightning_attention_cpu.h"
#include "../../../../infiniop/handle.h"
#include "../../../../utils.h"
#include "../../../../utils/custom_types.h"
#include <cmath>
#include <cstdint>
#include <vector>

namespace op::lightning_attention::cpu {

Descriptor::~Descriptor() {}

template <typename T>
static infiniStatus_t lightning_attention_cpu_impl(const LightningAttentionInfo &info,
                                                   T *out, T *initial_state,
                                                   const T *q, const T *k, const T *v,
                                                   const float *slope,
                                                   const void *init_idx_ptr,
                                                   const void *final_idx_ptr) {
    const size_t B = info.B;
    const size_t Tlen = info.T;
    const size_t H = info.H;
    const size_t D = info.D;

    const auto &out_s = info.out_strides;
    const auto &state_s = info.initial_state_strides;
    const auto &q_s = info.q_strides;
    const auto &k_s = info.k_strides;
    const auto &v_s = info.v_strides;
    const auto &slope_s = info.slope_strides;

    const auto read_index = [&](const void *ptr, size_t i) -> size_t {
        if (info.index_dtype == INFINI_DTYPE_I32) {
            return static_cast<size_t>(reinterpret_cast<const int32_t *>(ptr)[i]);
        }
        return static_cast<size_t>(reinterpret_cast<const int64_t *>(ptr)[i]);
    };

    // Per-request recurrent state [H, D, D], accumulated in fp32.
    std::vector<float> S(H * D * D, 0.0f);

    for (size_t b = 0; b < B; ++b) {
        const size_t state_row = read_index(init_idx_ptr, b);
        const size_t final_row = read_index(final_idx_ptr, b);

        // Load initial state.
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < D; ++i) {
                for (size_t j = 0; j < D; ++j) {
                    S[(h * D + i) * D + j] = utils::cast<float>(
                        initial_state[state_row * state_s[0] + h * state_s[1] + i * state_s[2] + j * state_s[3]]);
                }
            }
        }

        for (size_t t = 0; t < Tlen; ++t) {
            const size_t q_base = b * q_s[0] + t * q_s[1];
            const size_t k_base = b * k_s[0] + t * k_s[1];
            const size_t v_base = b * v_s[0] + t * v_s[1];
            const size_t out_base = b * out_s[0] + t * out_s[1];

            for (size_t h = 0; h < H; ++h) {
                const float ratio = std::exp(-slope[h * slope_s[0]]);
                const size_t head_base = (h * D) * D;
                const size_t q_h = q_base + h * q_s[2];
                const size_t k_h = k_base + h * k_s[2];
                const size_t v_h = v_base + h * v_s[2];
                const size_t out_h = out_base + h * out_s[2];

                // S' = ratio * S + k^T v
                for (size_t i = 0; i < D; ++i) {
                    const float k_i = utils::cast<float>(k[k_h + i * k_s[3]]);
                    float *S_i = S.data() + head_base + i * D;
                    for (size_t j = 0; j < D; ++j) {
                        const float v_j = utils::cast<float>(v[v_h + j * v_s[3]]);
                        S_i[j] = ratio * S_i[j] + k_i * v_j;
                    }
                }
                // o[j] = sum_i q[i] * S[i, j]
                for (size_t j = 0; j < D; ++j) {
                    float acc = 0.0f;
                    for (size_t i = 0; i < D; ++i) {
                        const float q_i = utils::cast<float>(q[q_h + i * q_s[3]]);
                        acc += q_i * S[head_base + i * D + j];
                    }
                    out[out_h + j * out_s[3]] = utils::cast<T>(acc);
                }
            }
        }

        // Write the final state back into the pool.
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < D; ++i) {
                for (size_t j = 0; j < D; ++j) {
                    initial_state[final_row * state_s[0] + h * state_s[1] + i * state_s[2] + j * state_s[3]] =
                        utils::cast<T>(S[(h * D + i) * D + j]);
                }
            }
        }
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t initial_state_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t slope_desc,
    infiniopTensorDescriptor_t initial_state_indices_desc,
    infiniopTensorDescriptor_t final_state_indices_desc) {
    auto result = LightningAttentionInfo::create(out_desc, initial_state_desc,
                                                 q_desc, k_desc, v_desc, slope_desc,
                                                 initial_state_indices_desc,
                                                 final_state_indices_desc);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out, void *initial_state,
    const void *q, const void *k, const void *v,
    const void *slope,
    const void *initial_state_indices,
    const void *final_state_indices,
    void *stream) const {
    if (_info.data_dtype == INFINI_DTYPE_F32) {
        return lightning_attention_cpu_impl<float>(
            _info, (float *)out, (float *)initial_state,
            (const float *)q, (const float *)k, (const float *)v,
            (const float *)slope, initial_state_indices, final_state_indices);
    }
    if (_info.data_dtype == INFINI_DTYPE_F16) {
        return lightning_attention_cpu_impl<fp16_t>(
            _info, (fp16_t *)out, (fp16_t *)initial_state,
            (const fp16_t *)q, (const fp16_t *)k, (const fp16_t *)v,
            (const float *)slope, initial_state_indices, final_state_indices);
    }
    if (_info.data_dtype == INFINI_DTYPE_BF16) {
        return lightning_attention_cpu_impl<bf16_t>(
            _info, (bf16_t *)out, (bf16_t *)initial_state,
            (const bf16_t *)q, (const bf16_t *)k, (const bf16_t *)v,
            (const float *)slope, initial_state_indices, final_state_indices);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::lightning_attention::cpu


