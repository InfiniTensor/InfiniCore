#include "infinicore/ops/moe_w4a8_marlin.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"

#include <functional>
#include <memory>
#include <stdexcept>

#if defined(ENABLE_HYGON_API) && defined(ENABLE_VENDOR_OPS)
namespace infinicore::op::moe_w4a8_marlin_impl::hygon {
void prepare_weight(Tensor output, const Tensor &input);
void align_routes(Tensor padded_sorted_token_ids,
                  Tensor expert_ids,
                  Tensor num_tokens_post_pad,
                  const Tensor &sorted_token_ids,
                  const Tensor &tokens_per_expert,
                  int64_t block_size,
                  int64_t routing_topk);
void run(Tensor output,
         const Tensor &input,
         const Tensor &marlin_weight,
         const Tensor &input_scale,
         const Tensor &weight_scale,
         std::optional<Tensor> topk_weights,
         const Tensor &padded_sorted_token_ids,
         const Tensor &expert_ids,
         const Tensor &num_tokens_post_pad,
         int64_t topk,
         int64_t routing_topk);
} // namespace infinicore::op::moe_w4a8_marlin_impl::hygon
#endif

namespace infinicore::op {
namespace {

class LambdaGraphOperator final : public graph::GraphOperator {
public:
    explicit LambdaGraphOperator(std::function<void()> runner)
        : runner_(std::move(runner)) {}

    void run() const override {
        runner_();
    }

private:
    std::function<void()> runner_;
};

void require_hygon_vendor(const Tensor &tensor, const char *name) {
#if defined(ENABLE_HYGON_API) && defined(ENABLE_VENDOR_OPS)
    if (tensor->device().getType() == Device::Type::HYGON) {
        return;
    }
#endif
    throw std::runtime_error(std::string(name)
                             + " has no registered implementation for this device");
}

} // namespace

void prepare_w4a8_marlin_weight_(Tensor output, const Tensor &input) {
    if (output->device() != input->device()
        || output->shape() != input->shape()
        || output->dtype() != DataType::I8
        || input->dtype() != DataType::I8
        || input->ndim() != 3
        || !output->is_contiguous()
        || !input->is_contiguous()) {
        throw std::runtime_error(
            "prepare_w4a8_marlin_weight expects contiguous int8 [E,N,K/2] tensors");
    }
    const size_t n = input->size(1);
    const size_t k = input->size(2) * 2;
    if ((n % 64) != 0 || (k % 32) != 0) {
        throw std::runtime_error(
            "prepare_w4a8_marlin_weight requires N%64==0 and K%32==0");
    }
    require_hygon_vendor(output, "prepare_w4a8_marlin_weight");
#if defined(ENABLE_HYGON_API) && defined(ENABLE_VENDOR_OPS)
    moe_w4a8_marlin_impl::hygon::prepare_weight(output, input);
#endif
}

void moe_align_block_size_from_counts_(
    Tensor padded_sorted_token_ids,
    Tensor expert_ids,
    Tensor num_tokens_post_pad,
    const Tensor &sorted_token_ids,
    const Tensor &tokens_per_expert,
    int64_t block_size,
    int64_t routing_topk) {
    if (block_size <= 0 || routing_topk <= 0 || routing_topk > 255
        || padded_sorted_token_ids->device() != sorted_token_ids->device()
        || expert_ids->device() != sorted_token_ids->device()
        || num_tokens_post_pad->device() != sorted_token_ids->device()
        || tokens_per_expert->device() != sorted_token_ids->device()
        || padded_sorted_token_ids->dtype() != DataType::I32
        || expert_ids->dtype() != DataType::I32
        || num_tokens_post_pad->dtype() != DataType::I32
        || sorted_token_ids->dtype() != DataType::I32
        || tokens_per_expert->dtype() != DataType::I32
        || padded_sorted_token_ids->numel()
               < sorted_token_ids->numel() * static_cast<size_t>(block_size)
        || expert_ids->numel() < sorted_token_ids->numel()
        || num_tokens_post_pad->numel() != 1
        || sorted_token_ids->numel()
                   % static_cast<size_t>(routing_topk)
               != 0
        || sorted_token_ids->numel() / static_cast<size_t>(routing_topk)
               >= (1u << 24)) {
        throw std::runtime_error("invalid moe_align_block_size_from_counts arguments");
    }
    if (context::isGraphRecording()) {
        context::addGraphOperator(std::make_shared<LambdaGraphOperator>(
            [padded_sorted_token_ids, expert_ids, num_tokens_post_pad,
             sorted_token_ids, tokens_per_expert, block_size, routing_topk] {
                moe_align_block_size_from_counts_(
                    padded_sorted_token_ids, expert_ids, num_tokens_post_pad,
                    sorted_token_ids, tokens_per_expert, block_size, routing_topk);
            }));
        return;
    }
    require_hygon_vendor(padded_sorted_token_ids,
                         "moe_align_block_size_from_counts");
#if defined(ENABLE_HYGON_API) && defined(ENABLE_VENDOR_OPS)
    moe_w4a8_marlin_impl::hygon::align_routes(
        padded_sorted_token_ids, expert_ids, num_tokens_post_pad,
        sorted_token_ids, tokens_per_expert, block_size, routing_topk);
#endif
}

void moe_w4a8_marlin_(
    Tensor output,
    const Tensor &input,
    const Tensor &marlin_weight,
    const Tensor &input_scale,
    const Tensor &weight_scale,
    std::optional<Tensor> topk_weights,
    const Tensor &padded_sorted_token_ids,
    const Tensor &expert_ids,
    const Tensor &num_tokens_post_pad,
    int64_t topk,
    int64_t routing_topk) {
    if (topk <= 0 || routing_topk <= 0
        || input->ndim() != 2 || output->ndim() != 2
        || marlin_weight->ndim() != 3
        || weight_scale->ndim() != 3
        || input->dtype() != DataType::I8
        || marlin_weight->dtype() != DataType::I8
        || input_scale->dtype() != DataType::F32
        || weight_scale->dtype() != DataType::F32
        || output->dtype() != DataType::BF16
        || output->size(0) != input->size(0) * static_cast<size_t>(topk)
        || output->size(1) != marlin_weight->size(1)
        || input_scale->shape()
               != std::vector<size_t>{input->size(0), 1}
        || weight_scale->shape()
               != std::vector<size_t>{marlin_weight->size(0),
                                      marlin_weight->size(1), 1}
        || (topk_weights
            && ((*topk_weights)->device() != output->device()
                || (*topk_weights)->dtype() != DataType::F32
                || !(*topk_weights)->is_contiguous()
                || (*topk_weights)->numel() != output->size(0)))
        || padded_sorted_token_ids->dtype() != DataType::I32
        || expert_ids->dtype() != DataType::I32
        || num_tokens_post_pad->dtype() != DataType::I32
        || num_tokens_post_pad->numel() != 1) {
        throw std::runtime_error("invalid moe_w4a8_marlin arguments");
    }
    if (context::isGraphRecording()) {
        context::addGraphOperator(std::make_shared<LambdaGraphOperator>(
            [output, input, marlin_weight, input_scale, weight_scale,
             topk_weights, padded_sorted_token_ids, expert_ids,
             num_tokens_post_pad, topk, routing_topk] {
                moe_w4a8_marlin_(
                    output, input, marlin_weight, input_scale, weight_scale,
                    topk_weights, padded_sorted_token_ids, expert_ids,
                    num_tokens_post_pad, topk, routing_topk);
            }));
        return;
    }
    require_hygon_vendor(output, "moe_w4a8_marlin");
#if defined(ENABLE_HYGON_API) && defined(ENABLE_VENDOR_OPS)
    moe_w4a8_marlin_impl::hygon::run(
        output, input, marlin_weight, input_scale, weight_scale, topk_weights,
        padded_sorted_token_ids, expert_ids, num_tokens_post_pad,
        topk, routing_topk);
#endif
}

} // namespace infinicore::op
