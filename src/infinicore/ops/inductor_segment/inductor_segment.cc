#include "infinicore/ops/inductor_segment.hpp"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#endif

#if defined(ENABLE_METAX_API)
#ifdef ENABLE_METAX_MC_API
#include <mcr/mc_runtime_api.h>
#else
#include <hcr/hc_runtime_api.h>
#endif
#elif defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API)
#include <cuda_runtime_api.h>
#endif

#include "../../utils.hpp"
#include "inductor_segment_registry.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/graph/capture_arena.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

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

/// Triton fused_moe_routed under MetaX stream capture (no aten body).
/// Distinct from INFINI_MOE_CAPTURE_SAFE (aten index_select+bmm).
/// Prefer ``INFINI_CUDAGRAPH_POLICY`` + decode phase over bare TRITON_CAPTURE;
/// explicit ``INFINI_MOE_TRITON_CAPTURE`` still wins when set.
bool moe_triton_capture_enabled() {
    return infinicore::context::moeTritonCaptureAllowed();
}

/// MoE may enter device capture (aten CAPTURE_SAFE or Triton TRITON_CAPTURE).
bool moe_device_capturable() {
    return moe_capture_safe_enabled() || moe_triton_capture_enabled();
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

/// Capture-safe device zero: ATen ``zero_()`` launches ``FillFunctor`` which HTC-faults
/// under MetaX ``hcStream`` capture on InfiniCore-owned buffers. Prefer stream memset
/// recorded into the capture graph (or skip when the whole tensor is overwritten).
void capture_safe_device_zero_(void *ptr, size_t nbytes) {
    if (ptr == nullptr || nbytes == 0) {
        return;
    }
    auto stream = infinicore::context::getStream();
#if defined(ENABLE_METAX_API) && defined(ENABLE_METAX_MC_API)
    auto st = mcMemsetAsync(ptr, 0, nbytes, static_cast<mcStream_t>(stream));
    if (st != mcSuccess) {
        throw std::runtime_error("capture_safe_device_zero_: mcMemsetAsync failed");
    }
#elif defined(ENABLE_METAX_API)
    auto st = hcMemsetAsync(ptr, 0, nbytes, static_cast<hcStream_t>(stream));
    if (st != hcSuccess) {
        throw std::runtime_error("capture_safe_device_zero_: hcMemsetAsync failed");
    }
#elif defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API)
    auto st = cudaMemsetAsync(ptr, 0, nbytes, static_cast<cudaStream_t>(stream));
    if (st != cudaSuccess) {
        throw std::runtime_error("capture_safe_device_zero_: cudaMemsetAsync failed");
    }
#else
    (void)stream;
    std::memset(ptr, 0, nbytes);
#endif
}

void capture_safe_aten_zero_(at::Tensor &t) {
    if (!t.defined() || t.nbytes() == 0) {
        return;
    }
    if (!infinicore::context::isDeviceStreamCapturing()) {
        t.zero_();
        return;
    }
    if (!t.is_contiguous()) {
        // Non-contiguous under capture: fall back to ATen (may HTC); prefer contiguous scratch.
        t.zero_();
        return;
    }
    capture_safe_device_zero_(t.data_ptr(), static_cast<size_t>(t.nbytes()));
}

/// Capture-safe dtype cast: MetaX ATen ``Tensor::to`` / ``_to_copy`` can launch
/// ``FillFunctor<long>`` (HTC) under ``hcStream`` capture. Prefer CaptureArena empty
/// + ``copy_`` (no fill kernel) when an arena is active; skip when dtype already matches.
at::Tensor capture_safe_to_dtype(const at::Tensor &src, at::ScalarType dtype) {
    if (!src.defined()) {
        return src;
    }
    if (src.scalar_type() == dtype) {
        return src;
    }
    if (!infinicore::context::isDeviceStreamCapturing()) {
        return src.to(dtype);
    }
    if (auto *arena = infinicore::graph::current_capture_arena()) {
        at::Tensor dst = arena->empty_aten(src.sizes(), src.options().dtype(dtype));
        dst.copy_(src);
        arena->retain(dst);
        return dst;
    }
    // No arena: last resort (may HTC on MetaX).
    return src.to(dtype);
}

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
    const int64_t copy_len = std::min(static_cast<int64_t>(valid_len), padded.size(1));
    // Decode M=1 with bucket==valid often overwrites the whole padded row — skip zero.
    if (copy_len < padded.size(1)) {
        capture_safe_aten_zero_(padded);
    }
    if (copy_len > 0) {
        auto prefix = pos.narrow(0, 0, copy_len).unsqueeze(0);
        padded.narrow(1, 0, copy_len).copy_(prefix.narrow(1, 0, copy_len));
    }
}

void restore_cuda_context(const infinicore::Device &device) {
    // Mid-capture / mid-record device switches trip "Switching device runtime during
    // graph recording may break the graph!" and poison MetaX stream capture replay.
    if (infinicore::context::isDeviceStreamCapturing()
        || infinicore::context::isGraphRecording()) {
        return;
    }
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
        if (auto *arena = infinicore::graph::current_capture_arena()) {
            scratch = arena->empty_aten(
                {batch, bucket_i, hidden_size},
                hidden.options());
        } else {
            scratch = at::empty({batch, bucket_i, hidden_size}, hidden.options());
        }
        capture_safe_aten_zero_(scratch);
    } else if (copy_len < bucket_i) {
        // Reuse: clear stale tail so padded tokens stay zeros.
        // Under capture, prefer full-tensor stream memset (narrow may be non-contiguous).
        if (infinicore::context::isDeviceStreamCapturing()) {
            capture_safe_aten_zero_(scratch);
        } else {
            scratch.narrow(1, copy_len, bucket_i - copy_len).zero_();
        }
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
    // Under capture: skip redundant ATen ``.to(kFloat)`` / Long→Int ``.to`` (FillFunctor HTC).
    auto scores = at::sigmoid(capture_safe_to_dtype(logits, at::kFloat));
    auto bias_f = capture_safe_to_dtype(bias, at::kFloat).reshape({1, -1});
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
        if (auto *arena = infinicore::graph::current_capture_arena()) {
            arena->retain(top2);
            arena->retain(group_scores);
            arena->retain(group_idx);
            arena->retain(keep);
            arena->retain(mask);
        }
    }
    at::Tensor topk_weights;
    at::Tensor topk_ids;
    if (auto *arena = infinicore::graph::current_capture_arena()) {
        // Arena out-buffers for topk — avoid ATen topk internal Long FillFunctor scratch.
        const int64_t t_tokens = choice.size(0);
        auto vals = arena->empty_aten({t_tokens, top_k}, choice.options());
        auto inds_long = arena->empty_aten(
            {t_tokens, top_k}, choice.options().dtype(at::kLong));
        at::topk_out(vals, inds_long, choice, top_k, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
        topk_weights = at::gather(scores, 1, inds_long);
        if (norm_topk_prob) {
            topk_weights =
                topk_weights / (topk_weights.sum(/*dim=*/-1, /*keepdim=*/true) + 1e-20);
        }
        topk_weights = topk_weights * routed_scaling_factor;
        topk_ids = capture_safe_to_dtype(inds_long, at::kInt);
        arena->retain(scores);
        arena->retain(bias_f);
        arena->retain(choice);
        arena->retain(vals);
        arena->retain(inds_long);
        arena->retain(topk_weights);
        arena->retain(topk_ids);
    } else {
        auto topk_ids_long = std::get<1>(at::topk(choice, top_k, /*dim=*/-1));
        topk_weights = at::gather(scores, 1, topk_ids_long);
        if (norm_topk_prob) {
            topk_weights =
                topk_weights / (topk_weights.sum(/*dim=*/-1, /*keepdim=*/true) + 1e-20);
        }
        topk_weights = topk_weights * routed_scaling_factor;
        topk_ids = topk_ids_long.to(at::kInt);
    }
    return {topk_weights, topk_ids};
}

/// Eager decode: D2H logits + CPU top-k + H2D (avoids MetaX launch tax on tiny E).
/// Only for ``n_group==1`` (MiniCPM5). Not capture-safe — caller must gate.
std::pair<at::Tensor, at::Tensor> grouped_sigmoid_topk_host(
    const at::Tensor &logits_gpu,
    const at::Tensor &bias_cpu_f32,
    int64_t top_k,
    bool norm_topk_prob,
    double routed_scaling_factor,
    at::ScalarType weight_dtype) {
    const int64_t t_tokens = logits_gpu.size(0);
    const int64_t n_experts = logits_gpu.size(1);
    auto logits_cpu = logits_gpu.to(at::Device(at::kCPU), /*non_blocking=*/false, /*copy=*/true)
                          .to(at::kFloat)
                          .contiguous();
    const float *logits = logits_cpu.data_ptr<float>();
    const float *bias = bias_cpu_f32.data_ptr<float>();

    struct HostScratch {
        std::vector<float> scores;
        std::vector<float> choice;
        std::vector<int32_t> idx;
        std::vector<float> w_host;
        std::vector<int32_t> id_host;
        at::Tensor pinned_w;
        at::Tensor pinned_ids;
    };
    static thread_local HostScratch hs;
    hs.scores.resize(static_cast<size_t>(n_experts));
    hs.choice.resize(static_cast<size_t>(n_experts));
    hs.idx.resize(static_cast<size_t>(n_experts));
    hs.w_host.resize(static_cast<size_t>(t_tokens * top_k));
    hs.id_host.resize(static_cast<size_t>(t_tokens * top_k));

    for (int64_t t = 0; t < t_tokens; ++t) {
        const float *row = logits + t * n_experts;
        for (int64_t e = 0; e < n_experts; ++e) {
            const float x = row[e];
            float s;
            if (x >= 0.0f) {
                const float z = std::exp(-x);
                s = 1.0f / (1.0f + z);
            } else {
                const float z = std::exp(x);
                s = z / (1.0f + z);
            }
            hs.scores[static_cast<size_t>(e)] = s;
            hs.choice[static_cast<size_t>(e)] = s + bias[e];
            hs.idx[static_cast<size_t>(e)] = static_cast<int32_t>(e);
        }
        const int64_t k = std::min(top_k, n_experts);
        auto nth = hs.idx.begin() + static_cast<std::ptrdiff_t>(k);
        std::nth_element(hs.idx.begin(), nth, hs.idx.end(), [&](int32_t a, int32_t b) {
            return hs.choice[static_cast<size_t>(a)] > hs.choice[static_cast<size_t>(b)];
        });
        // Stable-ish order among top-k for determinism vs GPU topk (sort by choice desc).
        std::sort(hs.idx.begin(), nth, [&](int32_t a, int32_t b) {
            return hs.choice[static_cast<size_t>(a)] > hs.choice[static_cast<size_t>(b)];
        });
        float denom = 0.0f;
        for (int64_t j = 0; j < k; ++j) {
            const int32_t id = hs.idx[static_cast<size_t>(j)];
            const float w = hs.scores[static_cast<size_t>(id)];
            hs.w_host[static_cast<size_t>(t * top_k + j)] = w;
            hs.id_host[static_cast<size_t>(t * top_k + j)] = id;
            denom += w;
        }
        if (norm_topk_prob) {
            const float inv = 1.0f / (denom + 1e-20f);
            for (int64_t j = 0; j < k; ++j) {
                hs.w_host[static_cast<size_t>(t * top_k + j)] *= inv;
            }
        }
        const float scale = static_cast<float>(routed_scaling_factor);
        for (int64_t j = 0; j < k; ++j) {
            hs.w_host[static_cast<size_t>(t * top_k + j)] *= scale;
        }
    }

    const int64_t n_out = t_tokens * top_k;
    if (!hs.pinned_w.defined() || hs.pinned_w.numel() < n_out) {
        hs.pinned_w = at::empty({n_out}, at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
        hs.pinned_ids = at::empty({n_out}, at::TensorOptions().dtype(at::kInt).pinned_memory(true));
    }
    std::memcpy(hs.pinned_w.data_ptr<float>(), hs.w_host.data(), sizeof(float) * static_cast<size_t>(n_out));
    std::memcpy(hs.pinned_ids.data_ptr<int32_t>(), hs.id_host.data(), sizeof(int32_t) * static_cast<size_t>(n_out));

    auto device = logits_gpu.device();
    // Blocking H2D: Triton must not race with incomplete copies.
    auto topk_w = hs.pinned_w.narrow(0, 0, n_out)
                      .to(device, /*non_blocking=*/false)
                      .to(weight_dtype)
                      .view({t_tokens, top_k});
    auto topk_ids = hs.pinned_ids.narrow(0, 0, n_out)
                        .to(device, /*non_blocking=*/false)
                        .view({t_tokens, top_k});
    return {topk_w, topk_ids};
}

at::Tensor silu_mlp_aten(
    const at::Tensor &x,
    const at::Tensor &w_gate_up,
    const at::Tensor &w_down) {
    auto gu = at::linear(x, w_gate_up);
    auto chunks = gu.chunk(2, /*dim=*/-1);
    auto mid = at::silu(chunks[0]) * chunks[1];
    auto out = at::linear(mid, w_down);
    if (auto *arena = infinicore::graph::current_capture_arena()) {
        arena->retain(gu);
        arena->retain(chunks[0]);
        arena->retain(chunks[1]);
        arena->retain(mid);
        arena->retain(out);
    }
    return out;
}

/// Reuse intermediate buffers for decode shared-expert MLP (Phase 7).
at::Tensor silu_mlp_aten_cached(
    const at::Tensor &x,
    const at::Tensor &w_gate_up,
    const at::Tensor &w_down) {
    struct Scratch {
        at::Tensor gu;
        at::Tensor mid;
        at::Tensor out;
        int64_t t = 0;
        int64_t h = 0;
        int64_t i2 = 0;
        void *w_gu = nullptr;
        void *w_d = nullptr;
        at::Device device = at::kCPU;
        at::ScalarType dtype = at::kFloat;
    };
    static thread_local Scratch sc;
    const int64_t t = x.size(0);
    const int64_t h = x.size(1);
    const int64_t i2 = w_gate_up.size(0);
    const bool need = !sc.gu.defined() || sc.t != t || sc.h != h || sc.i2 != i2
                      || sc.w_gu != w_gate_up.data_ptr() || sc.w_d != w_down.data_ptr()
                      || sc.device != x.device() || sc.dtype != x.scalar_type();
    if (need) {
        sc.gu = at::empty({t, i2}, x.options());
        sc.mid = at::empty({t, i2 / 2}, x.options());
        sc.out = at::empty({t, h}, x.options());
        sc.t = t;
        sc.h = h;
        sc.i2 = i2;
        sc.w_gu = w_gate_up.data_ptr();
        sc.w_d = w_down.data_ptr();
        sc.device = x.device();
        sc.dtype = x.scalar_type();
    }
    at::matmul_out(sc.gu, x, w_gate_up.t());
    auto gate = sc.gu.narrow(/*dim=*/1, /*start=*/0, /*length=*/i2 / 2);
    auto up = sc.gu.narrow(/*dim=*/1, /*start=*/i2 / 2, /*length=*/i2 / 2);
    at::silu_out(sc.mid, gate);
    sc.mid.mul_(up);
    at::matmul_out(sc.out, sc.mid, w_down.t());
    return sc.out;
}

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
/// Single-weight-set CUDAGraph (microbench): gate + GPU top-k + shared MLP.
/// Multi-layer serve keeps host top-k (27 graphed replays thrashed MetaX).
struct DecodeAuxGraph {
    std::unique_ptr<at::cuda::CUDAGraph> graph;
    at::Tensor static_x;
    at::Tensor logits_raw;
    at::Tensor logits_f;
    at::Tensor scores;
    at::Tensor choice;
    at::Tensor topk_w_f;
    at::Tensor topk_w;
    at::Tensor topk_ids;
    at::Tensor gu;
    at::Tensor mid;
    at::Tensor static_shared;
    at::Tensor gate_w;
    at::Tensor bias_f;
    at::Tensor shared_gu;
    at::Tensor shared_d;
    void *bias_src = nullptr;
    int64_t t = 0, h = 0, e = 0, i2 = 0, top_k = 0;
    bool norm_topk_prob = true;
    double routed_scaling = 3.66;
    bool ready = false;

    void ensure(
        const at::Tensor &x,
        const at::Tensor &gate_w_in,
        const at::Tensor &bias_fp32,
        const at::Tensor &sgu,
        const at::Tensor &sd,
        int64_t top_k_in,
        bool norm_in,
        double scale_in) {
        const int64_t tt = x.size(0), hh = x.size(1), ee = gate_w_in.size(0), ii2 = sgu.size(0);
        const bool need = !ready || t != tt || h != hh || e != ee || i2 != ii2
                          || top_k != top_k_in || norm_topk_prob != norm_in
                          || routed_scaling != scale_in
                          || gate_w.data_ptr() != gate_w_in.data_ptr()
                          || bias_src != bias_fp32.data_ptr()
                          || shared_gu.data_ptr() != sgu.data_ptr()
                          || shared_d.data_ptr() != sd.data_ptr()
                          || static_x.scalar_type() != x.scalar_type()
                          || static_x.device() != x.device();
        if (!need) return;
        ready = false;
        graph.reset();
        t = tt; h = hh; e = ee; i2 = ii2; top_k = top_k_in;
        norm_topk_prob = norm_in; routed_scaling = scale_in;
        gate_w = gate_w_in;
        bias_src = bias_fp32.data_ptr();
        bias_f = bias_fp32.reshape({1, -1}).contiguous();
        shared_gu = sgu; shared_d = sd;
        // Prefer aliasing the live input when stable (cpp_pad reuses one buffer).
        static_x = x;
        logits_raw = at::empty({t, e}, x.options());
        logits_f = at::empty({t, e}, x.options().dtype(at::kFloat));
        scores = at::empty({t, e}, x.options().dtype(at::kFloat));
        choice = at::empty({t, e}, x.options().dtype(at::kFloat));
        topk_w_f = at::empty({t, top_k}, x.options().dtype(at::kFloat));
        topk_w = at::empty({t, top_k}, x.options());
        topk_ids = at::empty({t, top_k}, x.options().dtype(at::kInt));
        gu = at::empty({t, i2}, x.options());
        mid = at::empty({t, i2 / 2}, x.options());
        static_shared = at::empty({t, h}, x.options());

        auto run_body = [&]() {
            at::matmul_out(logits_raw, static_x, gate_w.t());
            logits_f.copy_(logits_raw);
            at::sigmoid_out(scores, logits_f);
            at::add_out(choice, scores, bias_f);
            auto topk_out = at::topk(choice, top_k, /*dim=*/-1);
            auto ids = std::get<1>(topk_out);
            topk_ids.copy_(ids);
            at::gather_out(topk_w_f, scores, 1, ids);
            if (norm_topk_prob) {
                auto denom = topk_w_f.sum(/*dim=*/-1, /*keepdim=*/true) + 1e-20;
                topk_w_f.div_(denom);
            }
            topk_w_f.mul_(routed_scaling);
            topk_w.copy_(topk_w_f);
            at::matmul_out(gu, static_x, shared_gu.t());
            auto gate = gu.narrow(1, 0, i2 / 2);
            auto up = gu.narrow(1, i2 / 2, i2 / 2);
            at::silu_out(mid, gate);
            mid.mul_(up);
            at::matmul_out(static_shared, mid, shared_d.t());
        };

        auto cap_stream = at::cuda::getStreamFromPool(/*isHighPriority=*/false);
        {
            c10::cuda::CUDAStreamGuard guard(cap_stream);
            run_body();
            cap_stream.synchronize();
            graph = std::make_unique<at::cuda::CUDAGraph>();
            graph->capture_begin();
            run_body();
            graph->capture_end();
        }
        ready = true;
    }

    void replay(const at::Tensor &x) {
        // Skip H2D when the live buffer is the same storage (cpp_pad / stable decode).
        if (x.data_ptr() != static_x.data_ptr()) {
            static_x.copy_(x);
        }
        graph->replay();
    }
};

DecodeAuxGraph &decode_aux_graph_for(void *gate_ptr) {
    static thread_local std::unordered_map<void *, DecodeAuxGraph> cache;
    return cache[gate_ptr];
}

bool moe_aux_cudagraph_enabled(void *gate_ptr) {
    if (const char *v = std::getenv("INFINI_MOE_AUX_CUDAGRAPH")) {
        if (v[0] != '\0') {
            return !(std::string(v) == "0" || std::string(v) == "false");
        }
    }
    // Auto: only when a single gate weight set is ever seen (cpp_pad microbench).
    static thread_local void *only = nullptr;
    static thread_local bool multi = false;
    if (multi) return false;
    if (only == nullptr) {
        only = gate_ptr;
        return true;
    }
    if (only != gate_ptr) {
        multi = true;
        return false;
    }
    return true;
}
#endif

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
    // Cache fp32 bias (CPU+GPU) and a same-dtype gate view for cheap decode GEMM.
    struct GateCache {
        at::Tensor gate_fp32;
        at::Tensor gate_xdtype;
        at::Tensor bias_fp32;
        at::Tensor bias_cpu;
        void *gate_ptr = nullptr;
        void *bias_ptr = nullptr;
        at::ScalarType xdtype = at::kFloat;
    };
    static thread_local GateCache gate_cache;
    const bool capturing = infinicore::context::isGraphRecording()
                           || infinicore::context::isDeviceStreamCapturing();
    bool stream_capturing = false;
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
    stream_capturing =
        at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None;
#endif
    if (gate_cache.gate_ptr != weights.gate_weight.data_ptr()
        || gate_cache.bias_ptr != weights.e_score_correction_bias.data_ptr()
        || !gate_cache.gate_fp32.defined()
        || gate_cache.xdtype != x.scalar_type()) {
        // Under stream capture: prefer same-dtype views (no ATen ``.to`` FillFunctor,
        // no full-gate fp32 copy into the capture graph). Eager keeps ``.to`` + CPU bias.
        if (capturing || stream_capturing) {
            if (weights.gate_weight.scalar_type() == x.scalar_type()) {
                gate_cache.gate_xdtype = weights.gate_weight;
            } else {
                gate_cache.gate_xdtype =
                    capture_safe_to_dtype(weights.gate_weight, x.scalar_type());
            }
            // Capture path uses gate_xdtype only; keep gate_fp32 defined for cache key.
            gate_cache.gate_fp32 = gate_cache.gate_xdtype;
            if (weights.e_score_correction_bias.scalar_type() == at::kFloat) {
                gate_cache.bias_fp32 = weights.e_score_correction_bias;
            } else {
                gate_cache.bias_fp32 = capture_safe_to_dtype(
                    weights.e_score_correction_bias, at::kFloat);
            }
            gate_cache.bias_cpu = at::Tensor();
        } else {
            gate_cache.gate_fp32 = weights.gate_weight.to(at::kFloat).contiguous();
            gate_cache.gate_xdtype = weights.gate_weight.to(x.scalar_type()).contiguous();
            gate_cache.bias_fp32 = weights.e_score_correction_bias.to(at::kFloat).contiguous();
            gate_cache.bias_cpu = gate_cache.bias_fp32.to(at::Device(at::kCPU)).contiguous();
        }
        gate_cache.gate_ptr = weights.gate_weight.data_ptr();
        gate_cache.bias_ptr = weights.e_score_correction_bias.data_ptr();
        gate_cache.xdtype = x.scalar_type();
    }

    const int64_t t_tokens = x.size(0);
    // Host top-k does D2H/H2D + CPU work — illegal under hcStream capture.
    // Note: device-segment capture runs after GraphManager stops recording, so
    // isGraphRecording() is false; must also check isDeviceStreamCapturing().
    const bool use_host_topk = !capturing && !stream_capturing && t_tokens > 0
                               && t_tokens <= 16 && cfg.n_group == 1 && x.is_cuda();

    at::Tensor logits;
    at::Tensor topk_w;
    at::Tensor topk_ids;
    at::Tensor shared;
    at::Tensor routed;
    at::Tensor x_f;

    if (use_host_topk) {
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
        if (moe_aux_cudagraph_enabled(gate_cache.gate_xdtype.data_ptr())) {
            // Microbench / single-weight: CUDAGraph(gate+topk+shared) + Triton.
            auto &aux = decode_aux_graph_for(gate_cache.gate_xdtype.data_ptr());
            aux.ensure(
                x,
                gate_cache.gate_xdtype,
                gate_cache.bias_fp32,
                weights.shared_gate_up,
                weights.shared_down,
                cfg.top_k,
                cfg.norm_topk_prob,
                cfg.routed_scaling_factor);
            aux.replay(x);
            routed = call_fused_moe_routed(
                x, aux.topk_w, aux.topk_ids, weights.w_gate_up, weights.w_down);
            return routed.add(aux.static_shared);
        }
        // Multi-layer serve: host top-k + overlap shared on cached side stream.
        static thread_local at::cuda::CUDAStream shared_stream =
            at::cuda::getStreamFromPool(/*isHighPriority=*/false);
        at::Tensor shared_local;
        {
            c10::cuda::CUDAStreamGuard guard(shared_stream);
            shared_local = silu_mlp_aten_cached(x, weights.shared_gate_up, weights.shared_down);
        }
        logits = at::linear(x, gate_cache.gate_xdtype).to(at::kFloat);
        auto topk = grouped_sigmoid_topk_host(
            logits,
            gate_cache.bias_cpu,
            cfg.top_k,
            cfg.norm_topk_prob,
            cfg.routed_scaling_factor,
            x.scalar_type());
        topk_w = topk.first;
        topk_ids = topk.second;
        routed = call_fused_moe_routed(
            x, topk_w, topk_ids, weights.w_gate_up, weights.w_down);
        at::cuda::CUDAEvent done;
        done.record(shared_stream);
        done.block(at::cuda::getCurrentCUDAStream());
        return routed.add(shared_local);
#else
        logits = at::linear(x, gate_cache.gate_xdtype).to(at::kFloat);
        auto topk = grouped_sigmoid_topk_host(
            logits,
            gate_cache.bias_cpu,
            cfg.top_k,
            cfg.norm_topk_prob,
            cfg.routed_scaling_factor,
            x.scalar_type());
        topk_w = topk.first;
        topk_ids = topk.second;
        shared = silu_mlp_aten_cached(x, weights.shared_gate_up, weights.shared_down);
        routed = call_fused_moe_routed(
            x, topk_w, topk_ids, weights.w_gate_up, weights.w_down);
        return routed.add(shared);
#endif
    }

    // Capture / large-T: GPU top-k. Under stream capture keep gate/x dtype aligned
    // (no ``x.to(kFloat)`` FillFunctor); non-capture keeps fp32 router for AOTI parity.
    if (capturing || stream_capturing) {
        logits = at::linear(x, gate_cache.gate_xdtype);
        x_f = at::Tensor(); // unused under capture
    } else {
        x_f = x.to(at::kFloat);
        logits = at::linear(x_f, gate_cache.gate_fp32);
    }
    auto topk = grouped_sigmoid_topk_aten(
        logits,
        gate_cache.bias_fp32,
        cfg.top_k,
        cfg.n_group,
        cfg.topk_group,
        cfg.norm_topk_prob,
        cfg.routed_scaling_factor);
    topk_w = capture_safe_to_dtype(topk.first, x.scalar_type());
    topk_ids = topk.second;
    routed = call_fused_moe_routed(
        x, topk_w, topk_ids, weights.w_gate_up, weights.w_down);
    shared = capturing ? silu_mlp_aten(x, weights.shared_gate_up, weights.shared_down)
                       : silu_mlp_aten_cached(x, weights.shared_gate_up, weights.shared_down);
    auto y = routed + shared;
    if (auto *arena = infinicore::graph::current_capture_arena()) {
        if (x_f.defined()) {
            arena->retain(x_f);
        }
        arena->retain(logits);
        arena->retain(topk.first);
        arena->retain(topk_ids);
        arena->retain(topk_w);
        arena->retain(routed);
        arena->retain(shared);
        arena->retain(y);
    }
    return y;
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
    // Under Triton capture always use eager decode (AOTI+opaque failed instantiate).
    if (moe_eager_decode_enabled(valid_len) || moe_triton_capture_enabled()) {
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
                if (infinicore::context::isDeviceStreamCapturing()) {
                    if (!out_aten.is_contiguous()) {
                        throw std::runtime_error(
                            "InductorMoe: non-contiguous MoE out under capture; cannot stream-memset pad");
                    }
                    const size_t elem = static_cast<size_t>(out_aten.element_size());
                    const size_t row = static_cast<size_t>(hidden_size) * elem;
                    const size_t prefix = static_cast<size_t>(valid_len) * row;
                    const size_t total = static_cast<size_t>(out_aten.size(1)) * row;
                    for (int64_t b = 0; b < batch; ++b) {
                        auto *base = static_cast<char *>(out_aten.data_ptr())
                                     + static_cast<size_t>(b) * total;
                        capture_safe_device_zero_(base + prefix, total - prefix);
                    }
                } else {
                    out_aten.narrow(
                                1,
                                static_cast<int64_t>(valid_len),
                                out_aten.size(1) - static_cast<int64_t>(valid_len))
                        .zero_();
                }
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
    // CG-2 (INFINI_MOE_CAPTURE_SAFE=1): device capture with aten body under capture.
    // Triton-capture (INFINI_MOE_TRITON_CAPTURE=1): device capture with Triton body.
    host_break_ = !moe_device_capturable();
}

void InductorMoe::execute(
    const Tensor &hidden_states,
    Tensor &out,
    size_t layer_idx,
    size_t bucket) {
#ifdef ENABLE_ATEN
    // Eager fast path (same pattern as InductorSegment pre_attn).
    // Refuse Triton under a live device-capture stream unless capturable mode
    // (aten CAPTURE_SAFE or Triton TRITON_CAPTURE).
    if (!context::isGraphRecording()) {
        if (context::isDeviceStreamCapturing() && !moe_device_capturable()) {
            throw std::runtime_error(
                "InductorMoe: refusing AOTI+Triton MoE under hcStream capture "
                "(set INFINI_MOE_TRITON_CAPTURE=1 for Triton body, "
                "INFINI_MOE_CAPTURE_SAFE=1 for aten body, or use host-break)");
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
    if (infinicore::context::isDeviceStreamCapturing() && !moe_device_capturable()) {
        throw std::runtime_error(
            "InductorMoe::run: AOTI+Triton MoE must not run under hcStream capture "
            "(recorded as host_break; Graph splits device segments around MoE). "
            "Set INFINI_MOE_TRITON_CAPTURE=1 for Triton body, or "
            "INFINI_MOE_CAPTURE_SAFE=1 for aten capture-safe body.");
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
