#include "random_sample_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../info.h"
#include "infinicore.h"
#include <algorithm>
#include <cstdio>

namespace op::random_sample::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t result_desc,
    infiniopTensorDescriptor_t probs_desc) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto result = RandomSampleInfo::create(result_desc, probs_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(),
        0,
        nullptr,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::minWorkspaceSize() const {
    return _min_workspace_size;
}

template <typename DT>
struct ComputeType {
    using type = DT;
};

template <>
struct ComputeType<fp16_t> {
    using type = float;
};

template <>
struct ComputeType<bf16_t> {
    using type = float;
};

struct Algo {

    template <class Tidx, class Tval>
    static auto get(void const *ptr, size_t i) {
        return utils::cast<typename ComputeType<Tval>::type, Tval>(reinterpret_cast<Tval const *>(ptr)[i]);
    }

    template <class Tidx, class Tval>
    infiniStatus_t argmax(
        void *workspace, size_t workspace_size,
        void *result, void const *probs, size_t n,
        void *stream) {
        auto idx = reinterpret_cast<Tidx *>(result);
        *idx = 0;

        auto max_val = get<Tidx, Tval>(probs, 0);
        for (size_t i = 0; i < n; i++) {
            if (auto val = get<Tidx, Tval>(probs, i); val > max_val) {
                max_val = val;
                *idx = static_cast<Tidx>(i);
            }
        }

        return INFINI_STATUS_SUCCESS;
    }

    template <class Tidx, class Tval>
    infiniStatus_t random(
        void *workspace, size_t workspace_size,
        void *result, void const *probs, size_t n,
        float random_val, float topp, int topk, float temperature, float repetition_penalty,
        const uint32_t *previous_tokens, size_t previous_tokens_len,
        void *stream) {

        struct KVPair {
            Tidx idx;
            typename ComputeType<Tval>::type val;

            bool operator<(const KVPair &other) const {
                return val > other.val;
            }
        };

        auto idx = reinterpret_cast<Tidx *>(result);

        // Apply repetition penalty if needed
        std::vector<typename ComputeType<Tval>::type> penalized_probs(n);
        if (repetition_penalty != 1.0f) {
            // Initialize with original values
            for (size_t i = 0; i < n; i++) {
                penalized_probs[i] = get<Tidx, Tval>(probs, i);
            }

            // If previous_tokens are provided, only penalize those tokens (proper repetition penalty)
            // Otherwise, penalize all tokens (full-history penalty for backward compatibility)
            if (previous_tokens != nullptr && previous_tokens_len > 0) {
                // Proper repetition penalty: only penalize previously generated tokens
                for (size_t i = 0; i < previous_tokens_len; i++) {
                    uint32_t token_id = previous_tokens[i];
                    if (token_id < n) {
                        auto val = penalized_probs[token_id];
                        if (val > 0) {
                            penalized_probs[token_id] = val / repetition_penalty;
                        } else {
                            penalized_probs[token_id] = val * repetition_penalty;
                        }
                    }
                }
            } else {
                // Full-history penalty: penalize all tokens (backward compatibility)
                for (size_t i = 0; i < n; i++) {
                    auto val = penalized_probs[i];
                    if (val > 0) {
                        penalized_probs[i] = val / repetition_penalty;
                    } else {
                        penalized_probs[i] = val * repetition_penalty;
                    }
                }
            }
        }

        // build & sort
        std::vector<KVPair> pairs(n);
        for (size_t i = 0; i < n; i++) {
            if (repetition_penalty != 1.0f) {
                pairs[i] = {static_cast<Tidx>(i), penalized_probs[i]};
            } else {
                pairs[i] = {static_cast<Tidx>(i), get<Tidx, Tval>(probs, i)};
            }
        }
        std::sort(pairs.begin(), pairs.end());
        // softmax & sum
        auto const max_val = pairs[0].val;
        pairs[0].val = 1;
        for (size_t i = 1; i < n; i++) {
            pairs[i].val = pairs[i - 1].val + std::exp((pairs[i].val - max_val) / temperature);
        }
        // topk & topp & limit
        // Handle disabled topk (0 or -1 means consider all tokens, like vLLM)
        size_t effective_topk = (topk <= 0) ? n : std::min(static_cast<size_t>(topk), n);
        auto const pk = pairs[effective_topk - 1].val,
                   pp = pairs[n - 1].val * topp,
                   plimit = random_val * std::min(pk, pp);
        // sample
        for (size_t i = 0; i < n; i++) {
            if (plimit <= pairs[i].val) {
                *idx = pairs[i].idx;
                break;
            }
        }

        return INFINI_STATUS_SUCCESS;
    }
};

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    float random_val,
    float topp,
    int topk,
    float temperature,
    float repetition_penalty,
    const uint32_t *previous_tokens,
    size_t previous_tokens_len,
    void *stream) const {

    Calculate::calculate<Algo>(
        Algo{}, _info, workspace, workspace_size,
        result, probs,
        random_val, topp, topk, temperature, repetition_penalty,
        previous_tokens, previous_tokens_len,
        stream);

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::random_sample::cpu
