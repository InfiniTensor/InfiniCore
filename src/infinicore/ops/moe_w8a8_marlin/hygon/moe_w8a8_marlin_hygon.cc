#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/ops/moe_w8a8_marlin.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/lightop_adaptor.hpp"
#include "infinicore/ops/common/dispatcher.hpp"
#include "infinicore/ops/moe_sum.hpp"
#include "infinicore/ops/per_channel_quant_i8.hpp"

#include <ATen/ATen.h>
#include <c10/hip/HIPGuard.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::op {
namespace moe_w8a8_marlin_pack_impl {
using schema = Tensor (*)(const Tensor &);
common::OpDispatcher<schema> &dispatcher();
} // namespace moe_w8a8_marlin_pack_impl
} // namespace infinicore::op

namespace infinicore::op::moe_w8a8_marlin_impl::hygon {

namespace {

at::Tensor pack_one_expert(const at::Tensor &weight) {
    if (weight.dim() != 2) {
        throw std::runtime_error("w8a8 marlin pack expects each expert weight to be 2D");
    }
    if (weight.scalar_type() != at::kChar) {
        throw std::runtime_error("w8a8 marlin pack expects int8 weights");
    }
    auto q_w = weight.transpose(0, 1).contiguous();
    const int64_t size_k = q_w.size(0);
    const int64_t size_n = q_w.size(1);
    constexpr int64_t k_tile = 64;
    if (size_k % k_tile != 0) {
        throw std::runtime_error("w8a8 marlin pack requires K % 64 == 0");
    }
    auto packed = q_w.reshape({size_k / k_tile, k_tile, size_n})
        .transpose(1, 2)
        .reshape({size_k / k_tile, size_n * k_tile})
        .contiguous();
    return packed;
}

Tensor pack(const Tensor &weight) {
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
    auto weight_at = infinicore::adaptor::to_aten_tensor(weight);
    if (weight_at.dim() != 3) {
        throw std::runtime_error("w8a8 marlin pack expects weight shape [E, N, K]");
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
    graph::GraphTensor cache2_i8;
    graph::GraphTensor input_i8;
    graph::GraphTensor input_scale;
    graph::GraphTensor cache2_scale;
    graph::GraphTensor hidden_states;
    graph::GraphTensor w13_marlin;
    graph::GraphTensor w2_marlin;
    graph::GraphTensor w13_scale;
    graph::GraphTensor w2_scale;
    graph::GraphTensor topk_weights;
    graph::GraphTensor sorted_token_ids;
    graph::GraphTensor expert_ids;
    graph::GraphTensor num_tokens_post_padded;
    size_t top_k;
    int mode0;
    size_t block_size_m;
    int delta0;
    int mode1;
    int delta1;
};

void *plan(Tensor output,
           Tensor cache13,
           Tensor cache2_i8,
           Tensor input_i8,
           Tensor input_scale,
           Tensor cache2_scale,
           const Tensor &hidden_states,
           const Tensor &w13_marlin,
           const Tensor &w2_marlin,
           const Tensor &w13_scale,
           const Tensor &w2_scale,
           const Tensor &topk_weights,
           const Tensor &sorted_token_ids,
           const Tensor &expert_ids,
           const Tensor &num_tokens_post_padded,
           size_t top_k,
           int mode0,
           size_t block_size_m,
           int delta0,
           int mode1,
           int delta1) {
    infinicore::adaptor::lightop::preload_moe_w8a8_ops();
    if (mode0 == 1001 && delta0 == 1) {
        infinicore::adaptor::lightop::preload_moe_w8a8_marlin_asm();
    }
    return new PlannedMeta{
        graph::GraphTensor(output),
        graph::GraphTensor(cache13),
        graph::GraphTensor(cache2_i8),
        graph::GraphTensor(input_i8),
        graph::GraphTensor(input_scale),
        graph::GraphTensor(cache2_scale),
        graph::GraphTensor(hidden_states),
        graph::GraphTensor(w13_marlin),
        graph::GraphTensor(w2_marlin),
        graph::GraphTensor(w13_scale),
        graph::GraphTensor(w2_scale),
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(sorted_token_ids),
        graph::GraphTensor(expert_ids),
        graph::GraphTensor(num_tokens_post_padded),
        top_k,
        mode0,
        block_size_m,
        delta0,
        mode1,
        delta1};
}

void run(void *planned_meta) {
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    const auto hidden_shape = p->hidden_states->shape();
    const auto w13_shape = p->w13_marlin->shape();
    const auto w2_shape = p->w2_marlin->shape();
    if (hidden_shape.size() != 2 || w13_shape.size() != 3 || w2_shape.size() != 3) {
        throw std::runtime_error("w8a8 marlin fused dense expects hidden [M,K], w13/w2 [E,*,*]");
    }
    const size_t m = hidden_shape[0];
    const size_t k = hidden_shape[1];
    const size_t top_k = p->top_k;
    const size_t n = w2_shape[1] * 64;
    const size_t n2 = n * 2;
    if (w13_shape[1] * 64 != k ||
        w13_shape[2] != n2 * 64 ||
        w2_shape[2] != k * 64) {
        throw std::runtime_error("w8a8 marlin fused dense weight shape mismatch");
    }
    const infinicore::Shape input_i8_shape{m, k};
    const infinicore::Shape input_scale_shape{m, 1};
    const infinicore::Shape cache2_i8_shape{m * top_k, n};
    const infinicore::Shape cache2_scale_shape{m * top_k, 1};
    if (p->input_i8->shape() != input_i8_shape ||
        p->input_scale->shape() != input_scale_shape ||
        p->cache2_i8->shape() != cache2_i8_shape ||
        p->cache2_scale->shape() != cache2_scale_shape) {
        throw std::runtime_error("w8a8 marlin fused dense workspace shape mismatch");
    }
    if (p->input_i8->dtype() != infinicore::DataType::I8 ||
        p->cache2_i8->dtype() != infinicore::DataType::I8 ||
        p->input_scale->dtype() != infinicore::DataType::F32 ||
        p->cache2_scale->dtype() != infinicore::DataType::F32) {
        throw std::runtime_error("w8a8 marlin fused dense workspace dtype mismatch");
    }

    const bool output_need_copy_back = !p->output->is_contiguous();
    Tensor output_work_ic = output_need_copy_back ? p->output->contiguous() : Tensor(p->output);
    Tensor hidden_work_ic = p->hidden_states->is_contiguous() ? Tensor(p->hidden_states) : p->hidden_states->contiguous();

    const size_t cache1_numel = m * top_k * n2;
    const size_t cache3_numel = m * top_k * k;
    auto cache1_ic = p->cache13->narrow({{0, 0, cache1_numel}})->view({m, top_k, n2});
    auto cache1_2d_ic = cache1_ic->view({m * top_k, n2});
    auto cache3_ic = p->cache13->narrow({{0, 0, cache3_numel}})->view({m, top_k, k});

    infinicore::op::per_channel_quant_i8_(
        hidden_work_ic->view({m, k}),
        Tensor(p->input_i8),
        Tensor(p->input_scale));

    auto qhidden = infinicore::adaptor::to_aten_tensor(p->input_i8);
    auto hidden_scale = infinicore::adaptor::to_aten_tensor(p->input_scale);
    auto w13 = infinicore::adaptor::to_aten_tensor(p->w13_marlin);
    auto w2 = infinicore::adaptor::to_aten_tensor(p->w2_marlin);
    auto w13_scale = infinicore::adaptor::to_aten_tensor(p->w13_scale);
    auto w2_scale = infinicore::adaptor::to_aten_tensor(p->w2_scale);
    auto cache1 = infinicore::adaptor::to_aten_tensor(cache1_ic);
    auto cache1_2d = infinicore::adaptor::to_aten_tensor(cache1_2d_ic);
    auto qcache2 = infinicore::adaptor::to_aten_tensor(p->cache2_i8);
    auto cache2_scale = infinicore::adaptor::to_aten_tensor(p->cache2_scale);
    auto cache3 = infinicore::adaptor::to_aten_tensor(cache3_ic);
    auto topk_weights = infinicore::adaptor::to_aten_tensor(p->topk_weights);
    auto sorted_token_ids = infinicore::adaptor::to_aten_tensor(p->sorted_token_ids);
    auto expert_ids = infinicore::adaptor::to_aten_tensor(p->expert_ids);
    auto num_tokens_post_padded = infinicore::adaptor::to_aten_tensor(p->num_tokens_post_padded);
    if (p->block_size_m == 0) {
        throw std::runtime_error("w8a8 marlin fused dense requires nonzero block_size_m");
    }

    const size_t num_pairs = m * top_k;
    const size_t num_experts = w13_shape[0];
    const size_t vllm_max_tokens_padded =
        num_pairs < num_experts
            ? std::min(num_pairs * p->block_size_m,
                       num_pairs + num_experts * (p->block_size_m - 1))
            : num_pairs + num_experts * (p->block_size_m - 1);
    const size_t vllm_max_blocks =
        (vllm_max_tokens_padded + p->block_size_m - 1) / p->block_size_m;
    auto sorted_token_ids_lightop =
        vllm_max_tokens_padded < static_cast<size_t>(sorted_token_ids.size(0))
            ? sorted_token_ids.narrow(0, 0, static_cast<int64_t>(vllm_max_tokens_padded))
            : sorted_token_ids;
    auto expert_ids_lightop =
        vllm_max_blocks < static_cast<size_t>(expert_ids.size(0))
            ? expert_ids.narrow(0, 0, static_cast<int64_t>(vllm_max_blocks))
            : expert_ids;

    try {
        infinicore::adaptor::lightop::moe_gemm_marlin_w8a8(
            qhidden, w13, cache1, hidden_scale, w13_scale, std::nullopt,
            sorted_token_ids_lightop, expert_ids_lightop, num_tokens_post_padded,
            static_cast<int64_t>(top_k), p->mode0, p->delta0);
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Hygon W8A8 Marlin GEMM1 failed: ") + e.what());
    }

    std::optional<at::Tensor> num_local_tokens = std::nullopt;
    std::optional<at::Tensor> silu_expert_ids = std::nullopt;
    try {
        infinicore::adaptor::lightop::fuse_silu_mul_quant(
            cache1_2d,
            qcache2,
            cache2_scale,
            num_local_tokens,
            1,
            -1,
            silu_expert_ids);
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Hygon W8A8 Marlin fuse_silu_mul_quant failed: ") + e.what());
    }

    std::optional<at::Tensor> topk_weights_opt(topk_weights);
    try {
        infinicore::adaptor::lightop::moe_gemm_marlin_w8a8(
            qcache2, w2, cache3, cache2_scale, w2_scale, topk_weights_opt,
            sorted_token_ids_lightop, expert_ids_lightop, num_tokens_post_padded,
            1, p->mode1, p->delta1);
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Hygon W8A8 Marlin GEMM2 failed: ") + e.what());
    }

    auto output_work = infinicore::adaptor::to_aten_tensor(output_work_ic);
    infinicore::adaptor::lightop::moe_sum(
        cache3,
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
    infinicore::op::moe_w8a8_marlin_pack_impl::dispatcher().registerDevice(Device::Type::HYGON, &pack);
    MoeW8A8MarlinFusedDense::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MoeW8A8MarlinFusedDense::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MoeW8A8MarlinFusedDense::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::moe_w8a8_marlin_impl::hygon

#endif // ENABLE_HYGON_API && ENABLE_ATEN
