#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/ops/moe_w16a16_marlin.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/lightop_adaptor.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/common/dispatcher.hpp"
#include "infinicore/ops/moe_sum.hpp"
#include "infinicore/ops/silu_and_mul.hpp"

#include <ATen/ATen.h>
#include <c10/hip/HIPGuard.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op {
namespace moe_w16a16_marlin_pack_impl {
using schema = Tensor (*)(const Tensor &);
common::OpDispatcher<schema> &dispatcher();
} // namespace moe_w16a16_marlin_pack_impl
} // namespace infinicore::op

namespace infinicore::op::moe_w16a16_marlin_impl::hygon {

namespace {

std::vector<int64_t> weight_perm_data() {
    std::vector<int64_t> perm;
    perm.reserve(2048);
    for (int i = 0; i < 64; ++i) {
        for (int col = 0; col < 2; ++col) {
            const int cur_col = (i % 16) * 2 + col;
            for (int row = 0; row < 4; ++row) {
                const int cur_row = (i / 16) * 4 + row;
                perm.push_back(static_cast<int64_t>(cur_row * 32 + cur_col));
            }
        }
    }
    return perm;
}

at::Tensor pack_one_expert(const at::Tensor &weight) {
    if (weight.dim() != 2) {
        throw std::runtime_error("w16a16 marlin pack expects each expert weight to be 2D");
    }
    auto q_w = weight.transpose(0, 1).contiguous();
    const int64_t size_k = q_w.size(0);
    const int64_t size_n = q_w.size(1);
    if (size_k % 16 != 0 || size_n % 32 != 0) {
        throw std::runtime_error("w16a16 marlin pack requires K % 16 == 0 and N % 32 == 0");
    }
    const auto perm_vec = weight_perm_data();
    auto weight_perm = at::tensor(
        perm_vec,
        at::TensorOptions().dtype(at::kLong).device(q_w.device()));
    auto packed = q_w.reshape({size_k / 16, 16, size_n / 32, 32})
                      .permute({0, 2, 1, 3})
                      .reshape({size_k / 16, size_n * 16});
    packed = packed.reshape({-1, static_cast<int64_t>(perm_vec.size())})
                 .index_select(1, weight_perm)
                 .reshape({size_k / 16, size_n * 16})
                 .contiguous();
    return packed;
}

Tensor pack(const Tensor &weight) {
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
    auto weight_at = infinicore::adaptor::to_aten_tensor(weight);
    if (weight_at.dim() != 3) {
        throw std::runtime_error("w16a16 marlin pack expects weight shape [E, N, K]");
    }
    const int64_t num_experts = weight_at.size(0);
    auto packed0 = pack_one_expert(weight_at.select(0, 0));
    auto output_ic = Tensor::empty(
        {static_cast<size_t>(num_experts),
         static_cast<size_t>(packed0.size(0)),
         static_cast<size_t>(packed0.size(1))},
        weight->dtype(),
        weight->device());
    auto output_at = infinicore::adaptor::to_aten_tensor(output_ic);
    output_at.select(0, 0).copy_(packed0);
    for (int64_t expert = 1; expert < num_experts; ++expert) {
        auto packed = pack_one_expert(weight_at.select(0, expert));
        output_at.select(0, expert).copy_(packed);
    }
    return output_ic;
}

} // namespace

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor cache13;
    graph::GraphTensor cache2;
    graph::GraphTensor hidden_states;
    graph::GraphTensor w13_marlin;
    graph::GraphTensor w2_marlin;
    graph::GraphTensor topk_weights;
    graph::GraphTensor sorted_token_ids;
    graph::GraphTensor expert_ids;
    graph::GraphTensor num_tokens_post_padded;
    size_t top_k;
    int mode0;
    int delta0;
    int mode1;
    int delta1;
};

void *plan(Tensor output,
           Tensor cache13,
           Tensor cache2,
           const Tensor &hidden_states,
           const Tensor &w13_marlin,
           const Tensor &w2_marlin,
           const Tensor &topk_weights,
           const Tensor &sorted_token_ids,
           const Tensor &expert_ids,
           const Tensor &num_tokens_post_padded,
           size_t top_k,
           int mode0,
           int delta0,
           int mode1,
           int delta1) {
    infinicore::context::setDevice(hidden_states->device());
    const auto device = hidden_states->device();
    const auto same_device = [&](const Tensor &tensor) {
        return tensor && tensor->device() == device;
    };
    if (!same_device(output) || !same_device(cache13) || !same_device(cache2) ||
        !same_device(w13_marlin) || !same_device(w2_marlin) ||
        !same_device(topk_weights) || !same_device(sorted_token_ids) ||
        !same_device(expert_ids) || !same_device(num_tokens_post_padded)) {
        throw std::runtime_error("w16a16 marlin fused dense tensors must be on one device");
    }

    const bool bf16 = hidden_states->dtype() == infinicore::DataType::BF16;
    const bool direct_mode0 = bf16 && mode0 == 1000;
    const bool direct_mode1 = bf16 && mode1 == 1000;
    if ((direct_mode0 || direct_mode1) &&
        (!output->is_contiguous() || !cache13->is_contiguous() ||
         !cache2->is_contiguous() || !hidden_states->is_contiguous() ||
         !w13_marlin->is_contiguous() || !w2_marlin->is_contiguous() ||
         !topk_weights->is_contiguous() || !sorted_token_ids->is_contiguous() ||
         !expert_ids->is_contiguous() || !num_tokens_post_padded->is_contiguous())) {
        throw std::runtime_error("Hygon W16A16 Marlin mode 1000 requires contiguous tensors");
    }

    const bool preload_legacy_gemm = mode0 < 1000 || mode1 < 1000;
    const bool preload_legacy_asm =
        (mode0 >= 1000 && !direct_mode0) || (mode1 >= 1000 && !direct_mode1);
    infinicore::adaptor::lightop::preload_moe_w16a16_ops(
        preload_legacy_gemm, preload_legacy_asm);
    if (direct_mode0) {
        infinicore::adaptor::lightop::preload_moe_w16a16_marlin_asm(false);
    }
    if (direct_mode1) {
        infinicore::adaptor::lightop::preload_moe_w16a16_marlin_asm(true);
    }
    return new PlannedMeta{
        graph::GraphTensor(output), graph::GraphTensor(cache13), graph::GraphTensor(cache2),
        graph::GraphTensor(hidden_states), graph::GraphTensor(w13_marlin), graph::GraphTensor(w2_marlin),
        graph::GraphTensor(topk_weights), graph::GraphTensor(sorted_token_ids), graph::GraphTensor(expert_ids),
        graph::GraphTensor(num_tokens_post_padded), top_k, mode0, delta0, mode1, delta1};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    infinicore::context::setDevice(p->hidden_states->device());
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    const auto hidden_shape = p->hidden_states->shape();
    const auto w13_shape = p->w13_marlin->shape();
    const auto w2_shape = p->w2_marlin->shape();
    if (hidden_shape.size() != 2 || w13_shape.size() != 3 || w2_shape.size() != 3) {
        throw std::runtime_error("w16a16 marlin fused dense expects hidden [M,K], w13/w2 [E,*,*]");
    }
    const size_t m = hidden_shape[0];
    const size_t k = hidden_shape[1];
    const size_t n2 = w13_shape[2] / 16;
    const size_t n = n2 / 2;
    if (w13_shape[1] * 16 != k || w2_shape[1] * 16 != n || w2_shape[2] / 16 != k) {
        throw std::runtime_error("w16a16 marlin fused dense weight shape mismatch");
    }

    const bool uses_direct_mode1000 =
        p->hidden_states->dtype() == infinicore::DataType::BF16 &&
        (p->mode0 == 1000 || p->mode1 == 1000);
    if (uses_direct_mode1000 &&
        (!p->output->is_contiguous() || !p->hidden_states->is_contiguous())) {
        throw std::runtime_error("Hygon W16A16 Marlin mode 1000 requires contiguous input and output");
    }
    const bool output_need_copy_back =
        !uses_direct_mode1000 && !p->output->is_contiguous();
    Tensor output_work_ic = output_need_copy_back ? p->output->contiguous() : Tensor(p->output);
    Tensor hidden_work_ic =
        uses_direct_mode1000 || p->hidden_states->is_contiguous()
            ? Tensor(p->hidden_states)
            : p->hidden_states->contiguous();

    const size_t top_k = p->top_k;
    const size_t cache1_numel = m * top_k * n2;
    const size_t cache3_numel = m * top_k * k;
    const size_t cache2_numel = m * top_k * n;
    auto cache1_ic = p->cache13->narrow({{0, 0, cache1_numel}})->view({m * top_k, n2});
    auto cache3_ic = p->cache13->narrow({{0, 0, cache3_numel}})->view({m * top_k, k});
    auto cache2_ic = p->cache2->narrow({{0, 0, cache2_numel}})->view({m * top_k, n});

    auto hidden = infinicore::adaptor::to_aten_tensor(hidden_work_ic);
    auto w13 = infinicore::adaptor::to_aten_tensor(p->w13_marlin);
    auto w2 = infinicore::adaptor::to_aten_tensor(p->w2_marlin);
    auto cache1 = infinicore::adaptor::to_aten_tensor(cache1_ic);
    auto cache2 = infinicore::adaptor::to_aten_tensor(cache2_ic);
    auto cache3 = infinicore::adaptor::to_aten_tensor(cache3_ic);
    auto topk_weights = infinicore::adaptor::to_aten_tensor(p->topk_weights);
    auto sorted_token_ids = infinicore::adaptor::to_aten_tensor(p->sorted_token_ids);
    auto expert_ids = infinicore::adaptor::to_aten_tensor(p->expert_ids);
    auto num_tokens_post_padded = infinicore::adaptor::to_aten_tensor(p->num_tokens_post_padded);

    try {
        infinicore::adaptor::lightop::moe_gemm_marlin_w16a16(
            hidden, w13, cache1, std::nullopt, sorted_token_ids, expert_ids,
            num_tokens_post_padded, static_cast<int64_t>(top_k), p->mode0, p->delta0);
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Hygon W16A16 Marlin GEMM1 failed: ") + e.what());
    }

    infinicore::op::silu_and_mul_(cache2_ic, cache1_ic);

    std::optional<at::Tensor> topk_weights_opt(topk_weights);
    try {
        infinicore::adaptor::lightop::moe_gemm_marlin_w16a16(
            cache2, w2, cache3, topk_weights_opt, sorted_token_ids, expert_ids,
            num_tokens_post_padded, 1, p->mode1, p->delta1);
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Hygon W16A16 Marlin GEMM2 failed: ") + e.what());
    }

    auto cache3_reduce_ic = cache3_ic->view({m, top_k, k});
    auto cache3_reduce = infinicore::adaptor::to_aten_tensor(cache3_reduce_ic);
    auto output_work = infinicore::adaptor::to_aten_tensor(output_work_ic);
    infinicore::adaptor::lightop::moe_sum(
        cache3_reduce,
        output_work,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        1.0f,
        -1);

    if (output_need_copy_back) {
        p->output->copy_from(output_work_ic);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    if (!infinicore::adaptor::lightop::available()) {
        return false;
    }
    infinicore::op::moe_w16a16_marlin_pack_impl::dispatcher().registerDevice(Device::Type::HYGON, &pack);
    MoeW16A16MarlinFusedDense::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MoeW16A16MarlinFusedDense::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MoeW16A16MarlinFusedDense::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::moe_w16a16_marlin_impl::hygon

#endif // ENABLE_HYGON_API && ENABLE_ATEN
