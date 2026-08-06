#include "infinicore/ops/grouped_topk_vendor.hpp"
#include "../../utils.hpp"
#include "../vendor_ops/vendor_ops_dispatch.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"
#include "infinicore/ops/mul_scalar.hpp"
#include <functional>
#include <stdexcept>
namespace infinicore::op {
namespace {

class GroupedTopkVendorGraphOperator final : public graph::GraphOperator {
public:
    explicit GroupedTopkVendorGraphOperator(std::function<void()> runner)
        : runner_(std::move(runner)) {}

    void run() const override {
        runner_();
    }

private:
    std::function<void()> runner_;
};

} // namespace
void grouped_topk_vendor_(Tensor topk_weights, Tensor topk_ids, const Tensor &scores, int64_t num_expert_group, int64_t topk_group, bool renormalize, float routed_scaling_factor, const Tensor &bias, const std::string &scoring_func) {
    if (!topk_weights || !topk_ids || !scores) {
        throw std::runtime_error("grouped_topk_vendor expects non-empty topk_weights, topk_ids, scores");
    }
    if (bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_weights, topk_ids, scores, bias);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_weights, topk_ids, scores);
    }
    if (scores->ndim() != 2 || topk_weights->ndim() != 2 || topk_ids->ndim() != 2) {
        throw std::runtime_error("grouped_topk_vendor expects 2D tensors");
    }
    const auto tokens = scores->size(0), experts = scores->size(1), topk = topk_weights->size(1);
    if (topk_weights->size(0) != tokens || topk_ids->size(0) != tokens || topk_ids->size(1) != topk) {
        throw std::runtime_error("grouped_topk_vendor expects outputs (tokens, topk)");
    }
    if (num_expert_group < 1) {
        throw std::runtime_error("grouped_topk_vendor expects num_expert_group >= 1");
    }
    if (topk_group < 1 || topk_group > num_expert_group) {
        throw std::runtime_error("grouped_topk_vendor expects 1 <= topk_group <= num_expert_group");
    }
    if (experts % static_cast<size_t>(num_expert_group) != 0) {
        throw std::runtime_error("grouped_topk_vendor expects experts divisible by num_expert_group");
    }
    if (topk < 1 || topk > 32 || topk > experts) {
        throw std::runtime_error("grouped_topk_vendor expects topk in [1,32] and <= experts");
    }
    if (scores->dtype() != DataType::F16 && scores->dtype() != DataType::BF16
        && scores->dtype() != DataType::F32) {
        throw std::runtime_error("grouped_topk_vendor expects fp16/bfloat16/fp32 scores");
    }
    if (topk_weights->dtype() != DataType::F32) {
        throw std::runtime_error("grouped_topk_vendor expects topk_weights float32");
    }
    if (topk_ids->dtype() != DataType::I32 && topk_ids->dtype() != DataType::I64) {
        throw std::runtime_error("grouped_topk_vendor expects topk_ids int32/int64");
    }
    const bool valid_bias_dtype = !bias || bias->dtype() == scores->dtype()
                               || (scores->device().getType() == Device::Type::HYGON
                                   && bias->dtype() == DataType::F32);
    if (bias && (bias->numel() != experts || !valid_bias_dtype)) {
        throw std::runtime_error("grouped_topk_vendor expects bias shape (experts,) and same dtype as scores");
    }
    if (scoring_func != "softmax" && scoring_func != "sigmoid") {
        throw std::runtime_error("grouped_topk_vendor scoring_func must be softmax or sigmoid");
    }
    if (!topk_weights->is_contiguous() || !topk_ids->is_contiguous() || !scores->is_contiguous() || (bias && !bias->is_contiguous())) {
        throw std::runtime_error("grouped_topk_vendor expects contiguous tensors");
    }
    auto run_topk = [topk_weights, topk_ids, scores, num_expert_group,
                     topk_group, renormalize, bias, scoring_func] {
        auto kernel = vendor_ops::lookup(
            vendor_ops::grouped_topk_dispatcher(),
            scores->device().getType(),
            "grouped_topk_vendor");
        kernel(topk_weights, topk_ids, scores, num_expert_group, topk_group,
               renormalize, bias, scoring_func);
    };
    if (context::isGraphRecording()) {
        context::addGraphOperator(
            std::make_shared<GroupedTopkVendorGraphOperator>(run_topk));
    } else {
        run_topk();
    }
    if (routed_scaling_factor != 1.0f) {
        mul_scalar_(topk_weights, topk_weights, routed_scaling_factor);
    }
}
} // namespace infinicore::op
