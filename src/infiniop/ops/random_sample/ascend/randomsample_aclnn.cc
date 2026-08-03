#include "../../../devices/ascend/common_ascend.h"
#include "random_sample_aclnn.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace op::random_sample::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t probs;
    aclnnTensorDescriptor_t result;

    ~Opaque() {
        delete probs;
        delete result;
    }
};

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t result_desc,
    infiniopTensorDescriptor_t probs_desc) {
    auto handle = reinterpret_cast<device::ascend::Handle *>(handle_);
    auto result = RandomSampleInfo::create(result_desc, probs_desc);
    CHECK_RESULT(result);
    CHECK_DTYPE(result->dt_i, INFINI_DTYPE_I32, INFINI_DTYPE_I64);
    auto tresult = new aclnnTensorDescriptor(result_desc);
    auto tprobs = new aclnnTensorDescriptor(probs_desc);
    *desc_ptr
        = new Descriptor(
            result.take(),
            0,
            new Opaque{tprobs, tresult},
            handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::minWorkspaceSize() const {
    return _min_workspace_size;
}

infiniStatus_t
Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    float random_val,
    float topp,
    int topk,
    float temperature,
    void *stream) const {
    if (workspace_size < _min_workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (_info.n == 0) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Sampling needs a scalar result on the host. Stage the logits once and do
    // stable partial top-k here instead of copying FP32 logits back to the NPU,
    // running ACLNN TopK, and launching a second synchronization-prone kernel.
    CHECK_ACL(aclrtSynchronizeStream(static_cast<aclrtStream>(stream)));
    void *probs_host = nullptr;
    const auto probs_host_size = _info.n * infiniSizeOf(_info.dt_p);
    CHECK_ACL(aclrtMallocHost(&probs_host, probs_host_size));
    CHECK_ACL(aclrtMemcpy(
        probs_host, probs_host_size,
        probs, probs_host_size,
        ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> logits(_info.n);
    switch (_info.dt_p) {
    case INFINI_DTYPE_F16: {
        auto src = static_cast<const fp16_t *>(probs_host);
        for (size_t i = 0; i < _info.n; ++i) {
            logits[i] = _f16_to_f32(src[i]);
        }
        break;
    }
    case INFINI_DTYPE_BF16: {
        auto src = static_cast<const bf16_t *>(probs_host);
        for (size_t i = 0; i < _info.n; ++i) {
            logits[i] = _bf16_to_f32(src[i]);
        }
        break;
    }
    case INFINI_DTYPE_F32: {
        auto src = static_cast<const float *>(probs_host);
        std::copy(src, src + _info.n, logits.begin());
        break;
    }
    case INFINI_DTYPE_F64: {
        auto src = static_cast<const double *>(probs_host);
        for (size_t i = 0; i < _info.n; ++i) {
            logits[i] = static_cast<float>(src[i]);
        }
        break;
    }
    default:
        CHECK_ACL(aclrtFreeHost(probs_host));
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    CHECK_ACL(aclrtFreeHost(probs_host));

    const auto effective_topk = std::min(
        static_cast<size_t>(std::max(topk, 1)), _info.n);
    const bool do_sample
        = effective_topk > 1
       && temperature != 0.0f
       && topp != 0.0f
       && random_val != 0.0f;

    std::vector<size_t> indices(_info.n);
    std::iota(indices.begin(), indices.end(), 0);
    const auto greater_logit = [&logits](size_t lhs, size_t rhs) {
        if (logits[lhs] == logits[rhs]) {
            return lhs < rhs;
        }
        return logits[lhs] > logits[rhs];
    };

    size_t selected = 0;
    if (!do_sample) {
        selected = *std::max_element(
            indices.begin(), indices.end(),
            [&logits](size_t lhs, size_t rhs) {
                return logits[lhs] < logits[rhs];
            });
    } else {
        std::partial_sort(
            indices.begin(), indices.begin() + effective_topk, indices.end(),
            greater_logit);

        const double max_logit = logits[indices[0]];
        const double inv_temperature = 1.0 / static_cast<double>(temperature);
        const auto weight = [&logits, max_logit, inv_temperature](size_t index) {
            return std::exp(
                (static_cast<double>(logits[index]) - max_logit)
                * inv_temperature);
        };

        double total_mass = 0.0;
        for (size_t i = 0; i < _info.n; ++i) {
            total_mass += weight(i);
        }
        double topk_mass = 0.0;
        for (size_t i = 0; i < effective_topk; ++i) {
            topk_mass += weight(indices[i]);
        }

        const double limit = static_cast<double>(random_val)
                           * std::min(
                                 topk_mass,
                                 total_mass * static_cast<double>(topp));
        double cumulative = 0.0;
        selected = indices[effective_topk - 1];
        for (size_t i = 0; i < effective_topk; ++i) {
            cumulative += weight(indices[i]);
            if (limit <= cumulative) {
                selected = indices[i];
                break;
            }
        }
    }

    if (_info.dt_i == INFINI_DTYPE_I32) {
        const auto host_result = static_cast<int32_t>(selected);
        CHECK_ACL(aclrtMemcpy(
            result, sizeof(host_result),
            &host_result, sizeof(host_result),
            ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
        const auto host_result = static_cast<int64_t>(selected);
        CHECK_ACL(aclrtMemcpy(
            result, sizeof(host_result),
            &host_result, sizeof(host_result),
            ACL_MEMCPY_HOST_TO_DEVICE));
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::random_sample::ascend
