#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/ops/moe_align.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/lightop_adaptor.hpp"

#include <ATen/ATen.h>
#include <c10/hip/HIPGuard.h>

#include <optional>

namespace infinicore::op::moe_align_impl::infiniop {
void *plan(Tensor sorted_token_ids,
           Tensor expert_ids,
           Tensor num_tokens_post_padded,
           const Tensor &topk_ids,
           size_t num_experts,
           size_t block_size,
           bool pad_sorted_token_ids);
void run(void *planned_meta);
void cleanup(void **planned_meta_ptr);
} // namespace infinicore::op::moe_align_impl::infiniop
namespace infinicore::op::moe_align_impl::lightop_hygon {

struct PlannedMeta {
    graph::GraphTensor sorted_token_ids;
    graph::GraphTensor expert_ids;
    graph::GraphTensor num_tokens_post_padded;
    graph::GraphTensor topk_ids;
    size_t num_experts;
    size_t block_size;
    bool pad_sorted_token_ids;
    bool use_lightop;
    void *fallback;
};

void *plan(Tensor sorted_token_ids,
           Tensor expert_ids,
           Tensor num_tokens_post_padded,
           const Tensor &topk_ids,
           const size_t num_experts,
           const size_t block_size,
           const bool pad_sorted_token_ids) {
    const auto shape = topk_ids->shape();
    const bool use_lightop =
        shape.size() == 2 && shape[0] == 1 && shape[1] == 8 && num_experts == 128;
    if (use_lightop) {
        infinicore::adaptor::lightop::preload_moe_align();
    }

    void *fallback = nullptr;
    if (!use_lightop) {
        fallback = infinicore::op::moe_align_impl::infiniop::plan(
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_ids,
            num_experts,
            block_size,
            pad_sorted_token_ids);
    }

    return new PlannedMeta{
        graph::GraphTensor(sorted_token_ids),
        graph::GraphTensor(expert_ids),
        graph::GraphTensor(num_tokens_post_padded),
        graph::GraphTensor(topk_ids),
        num_experts,
        block_size,
        pad_sorted_token_ids,
        use_lightop,
        fallback};
}

void run(void *planned_meta) {
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    if (!p->use_lightop) {
        infinicore::op::moe_align_impl::infiniop::run(p->fallback);
        return;
    }

    auto topk_ids = infinicore::adaptor::to_aten_tensor(p->topk_ids);
    auto sorted_token_ids = infinicore::adaptor::to_aten_tensor(p->sorted_token_ids);
    auto expert_ids = infinicore::adaptor::to_aten_tensor(p->expert_ids);
    auto num_tokens_post_padded = infinicore::adaptor::to_aten_tensor(p->num_tokens_post_padded);

    const std::optional<at::Tensor> none = std::nullopt;
    infinicore::adaptor::lightop::moe_align_block_size(
        topk_ids,
        static_cast<int64_t>(p->num_experts),
        static_cast<int64_t>(p->block_size),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        none,
        none,
        none,
        false,
        p->pad_sorted_token_ids);
}

void cleanup(void **planned_meta_ptr) {
    auto *p = *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    if (p->fallback != nullptr) {
        infinicore::op::moe_align_impl::infiniop::cleanup(&p->fallback);
    }
    delete p;
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    if (!infinicore::adaptor::lightop::available()) {
        return false;
    }
    MoeAlign::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MoeAlign::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MoeAlign::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::moe_align_impl::lightop_hygon

#endif // ENABLE_HYGON_API && ENABLE_ATEN