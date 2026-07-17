#include "infinicore/ops/inductor_segment.hpp"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
#include <c10/cuda/CUDAGuard.h>
#endif

#include "../../utils.hpp"
#include "inductor_segment_registry.hpp"

#include "infinicore/context/context.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

size_t recording_valid_seq_len(size_t bucket) {
    if (const char *raw = std::getenv("INFINI_PIECEWISE_VALID_LEN")) {
        if (raw[0] != '\0') {
            return std::min(static_cast<size_t>(std::stoul(raw)), bucket);
        }
    }
    const size_t runtime_valid = infinicore::op::inductor_segment_impl::current_piecewise_valid_seq_len();
    if (runtime_valid > 0) {
        return std::min(runtime_valid, bucket);
    }
    return bucket;
}

/// CG-2: aten MoE under stream capture; Triton remains the eager path.
bool moe_capture_safe_enabled() {
    const char *v = std::getenv("INFINI_MOE_CAPTURE_SAFE");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

/// Decode-sized MoE: skip AOTI ``moe_B16/segment.pt2`` and call ``moe_block_eager``.
/// Default max tokens = 16 (covers MC=1 decode). Set ``INFINI_MOE_EAGER_DECODE_MAX=0``
/// to force the AOTI path.
size_t moe_eager_decode_max_tokens() {
    const char *v = std::getenv("INFINI_MOE_EAGER_DECODE_MAX");
    if (v == nullptr || v[0] == '\0') {
        return 16;
    }
    return static_cast<size_t>(std::stoul(v));
}

bool moe_eager_decode_enabled(size_t valid_len) {
    const size_t max_tok = moe_eager_decode_max_tokens();
    return max_tok > 0 && valid_len > 0 && valid_len <= max_tok;
}

size_t resolve_valid_seq_len(size_t bucket, const infinicore::Tensor &hidden_states) {
    if (infinicore::context::isGraphRecording()) {
        return recording_valid_seq_len(bucket);
    }
    // Eager decode tools / microbench: honor INFINI_PIECEWISE_VALID_LEN even when
    // hidden is already bucket-padded (shape[1]==bucket would otherwise hide valid=1).
    if (const char *raw = std::getenv("INFINI_PIECEWISE_VALID_LEN")) {
        if (raw[0] != '\0') {
            return std::min(static_cast<size_t>(std::stoul(raw)), bucket);
        }
    }
    const size_t runtime_valid = infinicore::op::inductor_segment_impl::current_piecewise_valid_seq_len();
    if (runtime_valid > 0) {
        return std::min(runtime_valid, bucket);
    }
    if (hidden_states) {
        const auto &shape = hidden_states->shape();
        if (shape.size() >= 2) {
            return std::min(static_cast<size_t>(shape[1]), bucket);
        }
    }
    return bucket;
}

} // namespace

#ifdef ENABLE_ATEN
#include "aot_package_runner.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/context/context.hpp"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#endif

namespace {

#ifdef ENABLE_ATEN
void fill_positions_padded_into(
    infinicore::Tensor &positions_padded,
    const infinicore::Tensor &positions,
    size_t bucket,
    size_t valid_len) {
    auto padded = infinicore::adaptor::to_aten_tensor(positions_padded);
    auto pos = infinicore::adaptor::to_aten_tensor(positions);
    if (pos.dim() == 2 && pos.size(0) == 1) {
        pos = pos.squeeze(0);
    }
    if (pos.dim() != 1) {
        throw std::runtime_error("InductorSegment: expected 1-D or [1, S] position_ids");
    }
    valid_len = std::min(valid_len, static_cast<size_t>(pos.size(0)));
    if (valid_len == 0) {
        valid_len = std::min(bucket, static_cast<size_t>(pos.size(0)));
    }
    padded.zero_();
    if (valid_len > 0) {
        auto prefix = pos.narrow(0, 0, static_cast<int64_t>(valid_len)).unsqueeze(0);
        const int64_t copy_len = std::min(static_cast<int64_t>(valid_len), padded.size(1));
        padded.narrow(1, 0, copy_len).copy_(prefix.narrow(1, 0, copy_len));
    }
}

void restore_cuda_context(const infinicore::Device &device) {
    infinicore::context::setDevice(device);
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
    c10::cuda::CUDAGuard guard(static_cast<int>(device.getIndex()));
#endif
}

auto as_contiguous_on_device(const at::Tensor &tensor) -> at::Tensor {
    if (tensor.is_contiguous()) {
        return tensor;
    }
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
    if (tensor.device().is_cuda()) {
        c10::cuda::CUDAGuard guard(static_cast<int>(tensor.device().index()));
        return tensor.contiguous();
    }
#endif
    return tensor.contiguous();
}

void copy_tensor_if_needed(infinicore::Tensor &dst, const at::Tensor &src) {
    const auto device = dst->device();
    restore_cuda_context(device);
    auto dst_aten = infinicore::adaptor::to_aten_tensor(dst);
    if (dst_aten.data_ptr() == src.data_ptr()) {
        return;
    }
    dst_aten.copy_(as_contiguous_on_device(src));
}

void run_pre_attn_segment(
    const infinicore::Tensor &positions,
    infinicore::Tensor &positions_padded,
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    infinicore::Tensor &q_rope,
    infinicore::Tensor &k_rope,
    infinicore::Tensor &v_rope,
    size_t layer_idx,
    size_t bucket,
    size_t valid_len) {
    const auto rank_device = hidden_states->device();
    restore_cuda_context(rank_device);

    bool layer_agnostic = false;
    auto &runner = infinicore::op::inductor_segment_impl::InductorSegmentRegistry::instance().runner(
        infinicore::op::PiecewiseInductorSegmentId::PreAttn,
        layer_idx,
        bucket,
        &layer_agnostic);

    fill_positions_padded_into(positions_padded, positions, bucket, valid_len);

    auto hidden = infinicore::adaptor::to_aten_tensor(hidden_states);
    auto res = infinicore::adaptor::to_aten_tensor(residual);
    auto pos = infinicore::adaptor::to_aten_tensor(positions_padded);

    std::vector<at::Tensor> inputs = {hidden, res, pos};
    if (layer_agnostic) {
        auto weights = infinicore::op::inductor_segment_impl::resolve_pre_attn_weights(layer_idx);
        inputs.push_back(weights.ln_weight);
        inputs.push_back(weights.q_weight);
        inputs.push_back(weights.k_weight);
        inputs.push_back(weights.v_weight);
        inputs.push_back(weights.q_norm_weight);
        inputs.push_back(weights.k_norm_weight);
    }

    std::vector<at::Tensor> outputs = runner.run(
        inputs,
        nullptr);

    if (outputs.size() < 5) {
        throw std::runtime_error(
            "InductorSegment pre_attn: expected 5 outputs, got "
            + std::to_string(outputs.size()));
    }

    copy_tensor_if_needed(hidden_states, outputs[0]);
    copy_tensor_if_needed(residual, outputs[1]);
    copy_tensor_if_needed(q_rope, outputs[2]);
    copy_tensor_if_needed(k_rope, outputs[3]);
    copy_tensor_if_needed(v_rope, outputs[4]);
    restore_cuda_context(rank_device);
    // Do not sync here: hcGraph capture/replay records InductorSegment as a graph op.
    // Eager callers (text_decoder_layer) sync after inductor_segment_ when not recording.
}

/// Reusable MoE pad scratch: skip ``at::zeros`` alloc when ``seq == bucket``;
/// otherwise reuse a thread-local buffer (zero only the unused tail).
thread_local at::Tensor g_moe_pad_scratch;

at::Tensor pad_moe_hidden_fast(const at::Tensor &hidden, size_t bucket, size_t valid_len) {
    if (hidden.dim() != 3) {
        throw std::runtime_error("InductorSegment moe: expected hidden [B, S, H]");
    }
    const int64_t batch = hidden.size(0);
    const int64_t seq = hidden.size(1);
    const int64_t hidden_size = hidden.size(2);
    // Already bucket-width (piecewise-padded decode or full-bucket prefill).
    if (static_cast<size_t>(seq) == bucket) {
        return hidden;
    }
    const int64_t bucket_i = static_cast<int64_t>(bucket);
    const int64_t copy_len = static_cast<int64_t>(std::min(valid_len, bucket));
    auto &scratch = g_moe_pad_scratch;
    const bool need_alloc = !scratch.defined()
                            || scratch.size(0) != batch
                            || scratch.size(1) != bucket_i
                            || scratch.size(2) != hidden_size
                            || scratch.device() != hidden.device()
                            || scratch.dtype() != hidden.dtype();
    if (need_alloc) {
        scratch = at::empty({batch, bucket_i, hidden_size}, hidden.options());
        scratch.zero_();
    } else if (copy_len < bucket_i) {
        // Reuse: clear stale tail so padded tokens stay zeros.
        scratch.narrow(1, copy_len, bucket_i - copy_len).zero_();
    }
    if (copy_len > 0) {
        scratch.narrow(1, 0, copy_len).copy_(hidden.narrow(1, 0, copy_len));
    }
    return scratch;
}

/// Phase 2: decode MoE without AOTI — C++ router/shared + Triton ``fused_moe_routed``.
at::Tensor call_fused_moe_routed(
    const at::Tensor &x,
    const at::Tensor &topk_w,
    const at::Tensor &topk_ids,
    const at::Tensor &w_gate_up,
    const at::Tensor &w_down) {
    static const c10::OperatorHandle op =
        c10::Dispatcher::singleton().findSchemaOrThrow("infinilm::fused_moe_routed", "");
    using Fn = at::Tensor(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor);
    return op.typed<Fn>().call(x, topk_w, topk_ids, w_gate_up, w_down);
}

/// MiniCPM5 grouped-sigmoid top-k (mirrors ``grouped_sigmoid_topk`` in Python).
std::pair<at::Tensor, at::Tensor> grouped_sigmoid_topk_aten(
    const at::Tensor &logits,
    const at::Tensor &bias,
    int64_t top_k,
    int64_t n_group,
    int64_t topk_group,
    bool norm_topk_prob,
    double routed_scaling_factor) {
    auto scores = at::sigmoid(logits.to(at::kFloat));
    auto bias_f = bias.to(at::kFloat).reshape({1, -1});
    auto choice = scores + bias_f;
    // MiniCPM5 default: n_group=1 → skip group mask (no-op path).
    if (n_group != 1) {
        const int64_t t_tokens = choice.size(0);
        const int64_t n_experts = choice.size(1);
        const int64_t experts_per_group = n_experts / n_group;
        auto choice_g = choice.view({t_tokens, n_group, experts_per_group});
        const int64_t k2 = std::min<int64_t>(2, experts_per_group);
        auto top2 = std::get<0>(at::topk(choice_g, k2, /*dim=*/-1));
        auto group_scores = top2.sum(/*dim=*/-1);
        auto group_idx = std::get<1>(at::topk(group_scores, topk_group, /*dim=*/-1));
        auto keep = at::zeros({t_tokens, n_group}, choice.options().dtype(at::kBool));
        keep.scatter_(1, group_idx, true);
        auto mask = keep.unsqueeze(-1)
                        .expand({t_tokens, n_group, experts_per_group})
                        .reshape({t_tokens, n_experts});
        choice = at::where(mask, choice, at::zeros_like(choice));
    }
    auto topk_ids = std::get<1>(at::topk(choice, top_k, /*dim=*/-1));
    auto topk_weights = at::gather(scores, 1, topk_ids);
    if (norm_topk_prob) {
        topk_weights = topk_weights / (topk_weights.sum(/*dim=*/-1, /*keepdim=*/true) + 1e-20);
    }
    topk_weights = topk_weights * routed_scaling_factor;
    // Match warmed Triton cubins (bf16 weights, int32 ids).
    return {topk_weights, topk_ids.to(at::kInt)};
}

at::Tensor silu_mlp_aten(
    const at::Tensor &x,
    const at::Tensor &w_gate_up,
    const at::Tensor &w_down) {
    auto gu = at::linear(x, w_gate_up);
    auto chunks = gu.chunk(2, /*dim=*/-1);
    return at::linear(at::silu(chunks[0]) * chunks[1], w_down);
}

struct MoeRoutingConfig {
    int64_t top_k = 16;
    int64_t n_group = 1;
    int64_t topk_group = 1;
    bool norm_topk_prob = true;
    double routed_scaling_factor = 3.66;
};

MoeRoutingConfig load_moe_routing_config() {
    MoeRoutingConfig cfg;
    if (const char *v = std::getenv("INFINI_MOE_TOP_K")) {
        if (v[0] != '\0') {
            cfg.top_k = static_cast<int64_t>(std::stoll(v));
        }
    }
    if (const char *v = std::getenv("INFINI_MOE_N_GROUP")) {
        if (v[0] != '\0') {
            cfg.n_group = static_cast<int64_t>(std::stoll(v));
        }
    }
    if (const char *v = std::getenv("INFINI_MOE_TOPK_GROUP")) {
        if (v[0] != '\0') {
            cfg.topk_group = static_cast<int64_t>(std::stoll(v));
        }
    }
    if (const char *v = std::getenv("INFINI_MOE_ROUTED_SCALING")) {
        if (v[0] != '\0') {
            cfg.routed_scaling_factor = std::stod(v);
        }
    }
    if (const char *v = std::getenv("INFINI_MOE_NORM_TOPK")) {
        if (v[0] != '\0') {
            cfg.norm_topk_prob = !(std::string(v) == "0" || std::string(v) == "false");
        }
    }
    return cfg;
}

at::Tensor run_moe_eager_decode(
    const at::Tensor &x,
    const infinicore::op::inductor_segment_impl::MoeExternalWeights &weights) {
    static const MoeRoutingConfig cfg = load_moe_routing_config();
    // Cache fp32 gate/bias views once (router always runs in fp32).
    struct GateFp32 {
        at::Tensor gate;
        at::Tensor bias;
        void *gate_ptr = nullptr;
        void *bias_ptr = nullptr;
    };
    static thread_local GateFp32 gate_fp32;
    if (gate_fp32.gate_ptr != weights.gate_weight.data_ptr()
        || gate_fp32.bias_ptr != weights.e_score_correction_bias.data_ptr()
        || !gate_fp32.gate.defined()) {
        gate_fp32.gate = weights.gate_weight.to(at::kFloat).contiguous();
        gate_fp32.bias = weights.e_score_correction_bias.to(at::kFloat).contiguous();
        gate_fp32.gate_ptr = weights.gate_weight.data_ptr();
        gate_fp32.bias_ptr = weights.e_score_correction_bias.data_ptr();
    }
    auto logits = at::linear(x.to(at::kFloat), gate_fp32.gate);
    auto topk = grouped_sigmoid_topk_aten(
        logits,
        gate_fp32.bias,
        cfg.top_k,
        cfg.n_group,
        cfg.topk_group,
        cfg.norm_topk_prob,
        cfg.routed_scaling_factor);
    auto topk_w = topk.first.to(x.dtype());
    auto routed = call_fused_moe_routed(
        x, topk_w, topk.second, weights.w_gate_up, weights.w_down);
    auto shared = silu_mlp_aten(x, weights.shared_gate_up, weights.shared_down);
    return routed + shared;
}

const infinicore::op::inductor_segment_impl::MoeExternalWeights &
cached_moe_weights(size_t layer_idx) {
    using W = infinicore::op::inductor_segment_impl::MoeExternalWeights;
    static thread_local std::unordered_map<size_t, W> cache;
    auto it = cache.find(layer_idx);
    if (it != cache.end()) {
        return it->second;
    }
    cache.emplace(layer_idx, infinicore::op::inductor_segment_impl::resolve_moe_weights(layer_idx));
    return cache.at(layer_idx);
}

void run_moe_segment(
    const infinicore::Tensor &hidden_states,
    infinicore::Tensor &out,
    size_t layer_idx,
    size_t bucket,
    size_t valid_len) {
    const auto rank_device = hidden_states->device();
    restore_cuda_context(rank_device);

    auto hidden = infinicore::adaptor::to_aten_tensor(hidden_states);
    if (hidden.dim() != 3) {
        throw std::runtime_error("InductorSegment moe: expected hidden [B, S, H]");
    }
    const int64_t batch = hidden.size(0);
    const int64_t seq = hidden.size(1);
    const int64_t hidden_size = hidden.size(2);
    // Prefer piecewise.valid_seq_len (decode=1) over padded shape[1]==bucket.
    valid_len = std::min(valid_len, bucket);
    if (valid_len == 0 || valid_len > static_cast<size_t>(seq)) {
        valid_len = std::min(static_cast<size_t>(seq), bucket);
    }

    // Decode / small-batch: bypass moe_B* AOTI — C++ router + Triton + shared.
    if (moe_eager_decode_enabled(valid_len)) {
        const auto &weights = cached_moe_weights(layer_idx);
        const int64_t t_tokens = batch * static_cast<int64_t>(valid_len);
        at::Tensor x = hidden.narrow(1, 0, static_cast<int64_t>(valid_len))
                           .reshape({t_tokens, hidden_size});
        if (!x.is_contiguous()) {
            x = x.contiguous();
        }
        at::Tensor y = run_moe_eager_decode(x, weights);
        restore_cuda_context(rank_device);
        auto out_aten = infinicore::adaptor::to_aten_tensor(out);
        at::Tensor y3 = y.view({batch, static_cast<int64_t>(valid_len), hidden_size});
        if (out_aten.sizes() == y3.sizes()) {
            out_aten.copy_(y3);
        } else {
            out_aten.narrow(1, 0, static_cast<int64_t>(valid_len)).copy_(y3);
            if (static_cast<size_t>(out_aten.size(1)) > valid_len) {
                out_aten.narrow(
                            1,
                            static_cast<int64_t>(valid_len),
                            out_aten.size(1) - static_cast<int64_t>(valid_len))
                    .zero_();
            }
        }
        restore_cuda_context(rank_device);
        return;
    }

    bool layer_agnostic = false;
    auto &runner = infinicore::op::inductor_segment_impl::InductorSegmentRegistry::instance().runner(
        infinicore::op::PiecewiseInductorSegmentId::Moe,
        layer_idx,
        bucket,
        &layer_agnostic);

    at::Tensor hidden_padded = pad_moe_hidden_fast(hidden, bucket, valid_len);

    std::vector<at::Tensor> inputs = {hidden_padded};
    if (layer_agnostic) {
        auto weights = infinicore::op::inductor_segment_impl::resolve_moe_weights(layer_idx);
        inputs.push_back(weights.gate_weight);
        inputs.push_back(weights.e_score_correction_bias);
        inputs.push_back(weights.w_gate_up);
        inputs.push_back(weights.w_down);
        inputs.push_back(weights.shared_gate_up);
        inputs.push_back(weights.shared_down);
    }

    std::vector<at::Tensor> outputs = runner.run(inputs, nullptr);
    if (outputs.empty()) {
        throw std::runtime_error("InductorSegment moe: expected >=1 output, got 0");
    }

    restore_cuda_context(rank_device);
    auto out_aten = infinicore::adaptor::to_aten_tensor(out);
    auto src = as_contiguous_on_device(outputs[0]);
    if (out_aten.sizes() == src.sizes()) {
        out_aten.copy_(src);
    } else {
        const int64_t copy_len = static_cast<int64_t>(std::min(valid_len, bucket));
        if (copy_len > 0) {
            out_aten.narrow(1, 0, copy_len).copy_(src.narrow(1, 0, copy_len));
        }
    }
    restore_cuda_context(rank_device);
}
#endif // ENABLE_ATEN

} // namespace

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(InductorSegment);

InductorSegment::InductorSegment(const Tensor &positions,
                                 Tensor &hidden_states,
                                 Tensor &residual,
                                 Tensor &q_rope,
                                 Tensor &k_rope,
                                 Tensor &v_rope,
                                 PiecewiseInductorSegmentId segment_id,
                                 size_t layer_idx,
                                 size_t bucket) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        positions, hidden_states, residual, q_rope, k_rope, v_rope);
    INFINICORE_GRAPH_OP_DISPATCH(
        hidden_states->device().getType(),
        positions,
        hidden_states,
        residual,
        q_rope,
        k_rope,
        v_rope,
        segment_id,
        layer_idx,
        bucket);
}

void InductorSegment::execute(const Tensor &positions,
                              Tensor &hidden_states,
                              Tensor &residual,
                              Tensor &q_rope,
                              Tensor &k_rope,
                              Tensor &v_rope,
                              PiecewiseInductorSegmentId segment_id,
                              size_t layer_idx,
                              size_t bucket) {
#ifdef ENABLE_ATEN
    if (segment_id == PiecewiseInductorSegmentId::PreAttn && !context::isGraphRecording()) {
        const size_t valid_len = resolve_valid_seq_len(bucket, hidden_states);
        auto positions_padded = infinicore::Tensor::zeros(
            {1, bucket},
            positions->dtype(),
            hidden_states->device());
        run_pre_attn_segment(
            positions,
            positions_padded,
            hidden_states,
            residual,
            q_rope,
            k_rope,
            v_rope,
            layer_idx,
            bucket,
            valid_len);
        return;
    }
#endif
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        InductorSegment,
        positions,
        hidden_states,
        residual,
        q_rope,
        k_rope,
        v_rope,
        segment_id,
        layer_idx,
        bucket);
}

void inductor_segment_(const Tensor &positions,
                       Tensor &hidden_states,
                       Tensor &residual,
                       Tensor &q_rope,
                       Tensor &k_rope,
                       Tensor &v_rope,
                       PiecewiseInductorSegmentId segment_id,
                       size_t layer_idx,
                       size_t bucket) {
    InductorSegment::execute(
        positions,
        hidden_states,
        residual,
        q_rope,
        k_rope,
        v_rope,
        segment_id,
        layer_idx,
        bucket);
}

void inductor_warmup_pre_attn_bucket(
    const Tensor &positions,
    Tensor &positions_padded,
    Tensor &hidden_states,
    Tensor &residual,
    size_t layer_idx,
    size_t bucket,
    size_t valid_len) {
#ifdef ENABLE_ATEN
    inductor_segment_impl::warmup_pre_attn(
        positions,
        positions_padded,
        hidden_states,
        residual,
        layer_idx,
        bucket,
        valid_len);
#else
    (void)positions;
    (void)positions_padded;
    (void)hidden_states;
    (void)residual;
    (void)layer_idx;
    (void)bucket;
    (void)valid_len;
    throw std::runtime_error(
        "inductor_warmup_pre_attn_bucket requires ENABLE_ATEN InfiniCore build");
#endif
}

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(InductorMoe);

InductorMoe::InductorMoe(
    const Tensor &hidden_states,
    Tensor &out,
    size_t layer_idx,
    size_t bucket) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(hidden_states, out);
    INFINICORE_GRAPH_OP_DISPATCH(
        hidden_states->device().getType(),
        hidden_states,
        out,
        layer_idx,
        bucket);
    // CG-1 default: Triton fused_moe_routed is not stream-capture-safe → host break.
    // CG-2 (INFINI_MOE_CAPTURE_SAFE=1): enter device capture; opaque op uses aten
    // under isDeviceStreamCapturing and Triton on the eager path.
    host_break_ = !moe_capture_safe_enabled();
}

void InductorMoe::execute(
    const Tensor &hidden_states,
    Tensor &out,
    size_t layer_idx,
    size_t bucket) {
#ifdef ENABLE_ATEN
    // Eager fast path (same pattern as InductorSegment pre_attn).
    // Refuse Triton under a live device-capture stream unless capture-safe mode
    // (aten body inside fused_moe_routed).
    if (!context::isGraphRecording()) {
        if (context::isDeviceStreamCapturing() && !moe_capture_safe_enabled()) {
            throw std::runtime_error(
                "InductorMoe: refusing AOTI+Triton MoE under hcStream capture "
                "(set INFINI_MOE_CAPTURE_SAFE=1 for aten body, or use host-break)");
        }
        const size_t valid_len = resolve_valid_seq_len(bucket, hidden_states);
        run_moe_segment(hidden_states, out, layer_idx, bucket, valid_len);
        return;
    }
#endif
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        InductorMoe,
        hidden_states,
        out,
        layer_idx,
        bucket);
}

void inductor_moe_(
    const Tensor &hidden_states,
    Tensor &out,
    size_t layer_idx,
    size_t bucket) {
#ifdef ENABLE_ATEN
    InductorMoe::execute(hidden_states, out, layer_idx, bucket);
#else
    (void)hidden_states;
    (void)out;
    (void)layer_idx;
    (void)bucket;
    throw std::runtime_error("inductor_moe_ requires ENABLE_ATEN InfiniCore build");
#endif
}

namespace inductor_segment_impl {

struct PlannedMeta {
    graph::GraphTensor positions;
    Tensor positions_padded;
    graph::GraphTensor hidden_states;
    graph::GraphTensor residual;
    graph::GraphTensor q_rope;
    graph::GraphTensor k_rope;
    graph::GraphTensor v_rope;
    PiecewiseInductorSegmentId segment_id;
    size_t layer_idx;
    size_t bucket;
    size_t valid_len;
};

struct MoePlannedMeta {
    graph::GraphTensor hidden_states;
    graph::GraphTensor out;
    size_t layer_idx;
    size_t bucket;
    size_t valid_len;
};

void *moe_plan(
    const Tensor &hidden_states,
    Tensor &out,
    size_t layer_idx,
    size_t bucket) {
    const size_t valid_len = infinicore::context::isGraphRecording()
        ? recording_valid_seq_len(bucket)
        : resolve_valid_seq_len(bucket, hidden_states);
    auto *meta = new MoePlannedMeta{
        graph::GraphTensor(hidden_states),
        graph::GraphTensor(out),
        layer_idx,
        bucket,
        valid_len,
    };
    return meta;
}

void moe_run(void *planned_meta) {
#ifdef ENABLE_ATEN
    if (infinicore::context::isDeviceStreamCapturing() && !moe_capture_safe_enabled()) {
        throw std::runtime_error(
            "InductorMoe::run: AOTI+Triton MoE must not run under hcStream capture "
            "(recorded as host_break; Graph splits device segments around MoE). "
            "Set INFINI_MOE_CAPTURE_SAFE=1 for aten capture-safe body.");
    }
    auto *meta = reinterpret_cast<MoePlannedMeta *>(planned_meta);
    if (meta == nullptr) {
        throw std::runtime_error("InductorMoe::run: null planned_meta");
    }
    Tensor hidden_states = meta->hidden_states->resume_from_blob_();
    Tensor out = meta->out->resume_from_blob_();
    run_moe_segment(
        hidden_states,
        out,
        meta->layer_idx,
        meta->bucket,
        meta->valid_len);
#else
    (void)planned_meta;
    throw std::runtime_error("InductorMoe requires ENABLE_ATEN build");
#endif
}

void moe_cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<MoePlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

void *plan(const Tensor &positions,
           Tensor &hidden_states,
           Tensor &residual,
           Tensor &q_rope,
           Tensor &k_rope,
           Tensor &v_rope,
           PiecewiseInductorSegmentId segment_id,
           size_t layer_idx,
           size_t bucket) {
    const size_t valid_len = infinicore::context::isGraphRecording()
        ? recording_valid_seq_len(bucket)
        : resolve_valid_seq_len(bucket, hidden_states);
    const auto device = hidden_states->device();
    // Never allocate via zeros during capture: pin_mode scratch is filled in run().
    infinicore::Tensor positions_padded = infinicore::context::isGraphRecording()
        ? infinicore::Tensor::empty({1, bucket}, positions->dtype(), device)
        : infinicore::Tensor::zeros({1, bucket}, positions->dtype(), device);
    auto *meta = new PlannedMeta{
        graph::GraphTensor(positions),
        positions_padded,
        graph::GraphTensor(hidden_states),
        graph::GraphTensor(residual),
        graph::GraphTensor(q_rope),
        graph::GraphTensor(k_rope),
        graph::GraphTensor(v_rope),
        segment_id,
        layer_idx,
        bucket,
        valid_len,
    };
    return meta;
}

void run(void *planned_meta) {
#ifdef ENABLE_ATEN
    auto *meta = reinterpret_cast<PlannedMeta *>(planned_meta);
    if (meta == nullptr) {
        throw std::runtime_error("InductorSegment::run: null planned_meta");
    }
    if (meta->segment_id != PiecewiseInductorSegmentId::PreAttn) {
        throw std::runtime_error(
            "InductorSegment: post_attn_cg AOT runtime not implemented yet");
    }
    // positions_padded is internal scratch (not a graph I/O edge).
    Tensor positions_padded = meta->positions_padded;
    Tensor hidden_states = meta->hidden_states->resume_from_blob_();
    Tensor residual = meta->residual->resume_from_blob_();
    Tensor q_rope = meta->q_rope->resume_from_blob_();
    Tensor k_rope = meta->k_rope->resume_from_blob_();
    Tensor v_rope = meta->v_rope->resume_from_blob_();
    run_pre_attn_segment(
        meta->positions->resume_from_blob_(),
        positions_padded,
        hidden_states,
        residual,
        q_rope,
        k_rope,
        v_rope,
        meta->layer_idx,
        meta->bucket,
        meta->valid_len);
#else
    (void)planned_meta;
    throw std::runtime_error("InductorSegment requires ENABLE_ATEN build");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

#ifdef ENABLE_ATEN
void warmup_pre_attn(
    const Tensor &positions,
    Tensor &positions_padded,
    const Tensor &hidden_states,
    const Tensor &residual,
    size_t layer_idx,
    size_t bucket,
    size_t valid_len) {
    const auto rank_device = hidden_states->device();
    restore_cuda_context(rank_device);

    bool layer_agnostic = false;
    auto &runner = InductorSegmentRegistry::instance().runner(
        PiecewiseInductorSegmentId::PreAttn, layer_idx, bucket, &layer_agnostic);
    fill_positions_padded_into(positions_padded, positions, bucket, valid_len);
    auto hidden = infinicore::adaptor::to_aten_tensor(hidden_states);
    auto res = infinicore::adaptor::to_aten_tensor(residual);
    auto pos = infinicore::adaptor::to_aten_tensor(positions_padded);
    std::vector<at::Tensor> inputs = {hidden, res, pos};
    if (layer_agnostic) {
        auto weights = resolve_pre_attn_weights(layer_idx);
        inputs.push_back(weights.ln_weight);
        inputs.push_back(weights.q_weight);
        inputs.push_back(weights.k_weight);
        inputs.push_back(weights.v_weight);
        inputs.push_back(weights.q_norm_weight);
        inputs.push_back(weights.k_norm_weight);
    }
    runner.warmup(inputs);
    restore_cuda_context(rank_device);
}
#endif

static bool registered_inductor_segment = []() {
    InductorSegment::plan_dispatcher().registerAll(&plan, false);
    InductorSegment::run_dispatcher().registerAll(&run, false);
    InductorSegment::cleanup_dispatcher().registerAll(&cleanup, false);
    return true;
}();
static bool registered_inductor_moe = []() {
    InductorMoe::plan_dispatcher().registerAll(&moe_plan, false);
    InductorMoe::run_dispatcher().registerAll(&moe_run, false);
    InductorMoe::cleanup_dispatcher().registerAll(&moe_cleanup, false);
    return true;
}();

namespace {

std::function<PreAttnExternalWeightTensors(size_t)> g_pre_attn_ic_resolver;

auto as_contiguous_aten(const infinicore::Tensor &tensor) -> at::Tensor {
    auto out = infinicore::adaptor::to_aten_tensor(tensor);
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
    if (out.device().is_cuda()) {
        c10::cuda::CUDAGuard guard(static_cast<int>(out.device().index()));
        return out.is_contiguous() ? out : out.contiguous();
    }
#endif
    return out.is_contiguous() ? out : out.contiguous();
}

PreAttnExternalWeights to_aten_weights(const PreAttnExternalWeightTensors &ic) {
    PreAttnExternalWeights out;
    out.ln_weight = as_contiguous_aten(ic.ln_weight);
    out.q_weight = as_contiguous_aten(ic.q_weight);
    out.k_weight = as_contiguous_aten(ic.k_weight);
    out.v_weight = as_contiguous_aten(ic.v_weight);
    out.q_norm_weight = as_contiguous_aten(ic.q_norm_weight);
    out.k_norm_weight = as_contiguous_aten(ic.k_norm_weight);
    return out;
}

std::function<MoeExternalWeightTensors(size_t)> g_moe_ic_resolver;

MoeExternalWeights to_aten_moe_weights(const MoeExternalWeightTensors &ic) {
    MoeExternalWeights out;
    out.gate_weight = as_contiguous_aten(ic.gate_weight);
    out.e_score_correction_bias = as_contiguous_aten(ic.e_score_correction_bias);
    out.w_gate_up = as_contiguous_aten(ic.w_gate_up);
    out.w_down = as_contiguous_aten(ic.w_down);
    out.shared_gate_up = as_contiguous_aten(ic.shared_gate_up);
    out.shared_down = as_contiguous_aten(ic.shared_down);
    return out;
}

} // namespace

void set_pre_attn_weight_resolver(
    std::function<PreAttnExternalWeightTensors(size_t layer_idx)> resolver) {
    g_pre_attn_ic_resolver = std::move(resolver);
    if (!g_pre_attn_ic_resolver) {
        inductor_segment_impl::set_pre_attn_aten_weight_resolver(nullptr);
        return;
    }
    inductor_segment_impl::set_pre_attn_aten_weight_resolver([](size_t layer_idx) {
        if (!g_pre_attn_ic_resolver) {
            throw std::runtime_error("InductorSegment: pre_attn weight resolver cleared");
        }
        return to_aten_weights(g_pre_attn_ic_resolver(layer_idx));
    });
}

void clear_pre_attn_weight_resolver() {
    g_pre_attn_ic_resolver = nullptr;
    inductor_segment_impl::set_pre_attn_aten_weight_resolver(nullptr);
}

void set_moe_weight_resolver(
    std::function<MoeExternalWeightTensors(size_t layer_idx)> resolver) {
    g_moe_ic_resolver = std::move(resolver);
    if (!g_moe_ic_resolver) {
        inductor_segment_impl::set_moe_aten_weight_resolver(nullptr);
        return;
    }
    inductor_segment_impl::set_moe_aten_weight_resolver([](size_t layer_idx) {
        if (!g_moe_ic_resolver) {
            throw std::runtime_error("InductorSegment: moe weight resolver cleared");
        }
        return to_aten_moe_weights(g_moe_ic_resolver(layer_idx));
    });
}

void clear_moe_weight_resolver() {
    g_moe_ic_resolver = nullptr;
    inductor_segment_impl::set_moe_aten_weight_resolver(nullptr);
}

bool has_moe_weight_resolver() {
    return static_cast<bool>(g_moe_ic_resolver);
}

} // namespace inductor_segment_impl

} // namespace infinicore::op
