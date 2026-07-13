#include "infinicore/ops/inductor_segment.hpp"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
#include <c10/cuda/CUDAGuard.h>
#endif

#include "../../utils.hpp"
#include "inductor_segment_registry.hpp"

#include "infinicore/context/context.hpp"

#include <sstream>

namespace {

size_t resolve_valid_seq_len(size_t bucket, const infinicore::Tensor &hidden_states) {
    const size_t runtime_valid = infinicore::op::inductor_segment_impl::current_piecewise_valid_seq_len();
    if (runtime_valid > 0) {
        return std::min(runtime_valid, bucket);
    }
    if (hidden_states->shape().size() >= 2) {
        return std::min(static_cast<size_t>(hidden_states->shape()[1]), bucket);
    }
    return bucket;
}

} // namespace

#ifdef ENABLE_ATEN
#include "aot_package_runner.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/context/context.hpp"
#endif

#include <fstream>
#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace {

// #region agent log
void agent_debug_log(
    const char *location,
    const char *message,
    const char *hypothesis_id,
    const std::string &data_json) {
    const char *path = std::getenv("INFINI_DEBUG_SESSION_LOG");
    if (path == nullptr || path[0] == '\0') {
        path = std::getenv("INFINI_AGENT_DEBUG_LOG");
    }
    if (path == nullptr || path[0] == '\0') {
        path = std::getenv("INFINI_DEBUG_LOG");
    }
    // Container mounts worktree at /workspace; host mirror is worktree/.cursor/.
    const char *paths[] = {
        path,
        "/workspace/.cursor/debug-685da7.log",
        nullptr,
    };
    const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
    const std::string line = std::string("{\"sessionId\":\"685da7\",\"location\":\"")
        + location + "\",\"message\":\"" + message + "\",\"hypothesisId\":\""
        + hypothesis_id + "\",\"data\":" + data_json + ",\"timestamp\":" + std::to_string(ts)
        + "}\n";
    // Also emit to stderr so docker logs capture the last step before a segfault.
    std::fprintf(stderr, "[agent_debug] %s %s %s\n", location, message, data_json.c_str());
    std::fflush(stderr);
    for (int i = 0; paths[i] != nullptr; ++i) {
        if (paths[i] == nullptr || paths[i][0] == '\0') {
            continue;
        }
        std::ofstream out(paths[i], std::ios::app);
        if (!out) {
            continue;
        }
        out << line;
        out.flush();
    }
}
// #endregion

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

    // #region agent log
    // Never sync GPU->CPU during hcGraph capture (e.g. .item()).
    if (bucket == 4 && valid_len > 0 && !infinicore::context::isGraphRecording()) {
        const int64_t trace_len = std::min<int64_t>(static_cast<int64_t>(valid_len), 4);
        std::ostringstream pos_json;
        pos_json << "[";
        for (int64_t i = 0; i < trace_len; ++i) {
            if (i > 0) {
                pos_json << ",";
            }
            pos_json << padded.index({0, i}).item<int64_t>();
        }
        pos_json << "]";
        agent_debug_log(
            "inductor_segment.cc:fill_positions_padded_into",
            "b4_padded_positions",
            "B4-POS",
            std::string("{\"bucket\":") + std::to_string(bucket) + ",\"valid_len\":" +
                std::to_string(valid_len) + ",\"positions\":" + pos_json.str() + "}");
    }
    // #endregion
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

    // #region agent log
    if (layer_idx == 0) {
        agent_debug_log(
            "inductor_segment.cc:run_pre_attn_segment",
            "aot_valid_len",
            "H3",
            std::string("{\"layer\":") + std::to_string(layer_idx) + ",\"bucket\":" +
                std::to_string(bucket) + ",\"valid_len\":" + std::to_string(valid_len) + "}");
    }
    // #endregion

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

    // #region agent log
    agent_debug_log(
        "inductor_segment.cc:run_pre_attn_segment",
        "aot_run_inputs",
        "H2",
        std::string("{\"stream_ptr\":")
            + std::to_string(reinterpret_cast<uintptr_t>(infinicore::context::getStream()))
            + ",\"pos_padded_ptr\":"
            + std::to_string(reinterpret_cast<uintptr_t>(pos.data_ptr())) + "}");
    // #endregion

    agent_debug_log(
        "inductor_segment.cc:run_pre_attn_segment",
        "before_aot_run",
        "H6",
        std::string("{\"n_inputs\":") + std::to_string(inputs.size())
            + ",\"layer_agnostic\":" + (layer_agnostic ? "true" : "false") + "}");

    std::vector<at::Tensor> outputs = runner.run(
        inputs,
        nullptr);

    agent_debug_log(
        "inductor_segment.cc:run_pre_attn_segment",
        "after_aot_run",
        "H6",
        std::string("{\"n_outputs\":") + std::to_string(outputs.size()) + "}");

    if (outputs.size() < 5) {
        throw std::runtime_error(
            "InductorSegment pre_attn: expected 5 outputs, got "
            + std::to_string(outputs.size()));
    }

    copy_tensor_if_needed(hidden_states, outputs[0]);
    copy_tensor_if_needed(residual, outputs[1]);
    agent_debug_log(
        "inductor_segment.cc:run_pre_attn_segment",
        "after_copy_hr",
        "H6",
        "{}");
    copy_tensor_if_needed(q_rope, outputs[2]);
    copy_tensor_if_needed(k_rope, outputs[3]);
    copy_tensor_if_needed(v_rope, outputs[4]);
    agent_debug_log(
        "inductor_segment.cc:run_pre_attn_segment",
        "after_copy_qkv",
        "H6",
        "{}");
    restore_cuda_context(rank_device);
    // Do not sync here: hcGraph capture/replay records InductorSegment as a graph op.
    // Eager callers (text_decoder_layer) sync after inductor_segment_ when not recording.
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
    // #region agent log
    agent_debug_log(
        "inductor_segment.cc:execute",
        "execute_enter",
        "H4",
        std::string("{\"recording\":")
            + (context::isGraphRecording() ? "true" : "false")
            + ",\"bucket\":" + std::to_string(bucket)
            + ",\"layer\":" + std::to_string(layer_idx) + "}");
    // #endregion
#ifdef ENABLE_ATEN
    if (segment_id == PiecewiseInductorSegmentId::PreAttn && !context::isGraphRecording()) {
        // #region agent log
        agent_debug_log(
            "inductor_segment.cc:execute",
            "eager_bypass",
            "H4",
            "{\"path\":\"eager\"}");
        // #endregion
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
    // #region agent log
    agent_debug_log(
        "inductor_segment.cc:execute",
        "record_or_run",
        "H4",
        "{\"path\":\"graph_op\"}");
    // #endregion
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

void *plan(const Tensor &positions,
           Tensor &hidden_states,
           Tensor &residual,
           Tensor &q_rope,
           Tensor &k_rope,
           Tensor &v_rope,
           PiecewiseInductorSegmentId segment_id,
           size_t layer_idx,
           size_t bucket) {
    agent_debug_log(
        "inductor_segment.cc:plan",
        "plan_begin",
        "H3",
        std::string("{\"bucket\":") + std::to_string(bucket) + ",\"layer\":" + std::to_string(layer_idx)
            + ",\"recording\":" + (infinicore::context::isGraphRecording() ? "true" : "false") + "}");
    const size_t valid_len = resolve_valid_seq_len(bucket, hidden_states);
    agent_debug_log(
        "inductor_segment.cc:plan",
        "after_valid_len",
        "H3",
        std::string("{\"valid_len\":") + std::to_string(valid_len) + "}");
    const auto device = hidden_states->device();
    agent_debug_log(
        "inductor_segment.cc:plan",
        "before_zeros",
        "H1",
        std::string("{\"device_index\":") + std::to_string(device.getIndex()) + "}");
    auto positions_padded = infinicore::Tensor::zeros(
        {1, bucket},
        positions->dtype(),
        device);
    agent_debug_log(
        "inductor_segment.cc:plan",
        "after_zeros",
        "H1",
        std::string("{\"valid_len\":") + std::to_string(valid_len) + "}");
    // Build GraphTensors one-by-one to isolate to_blob_ crashes on from_torch memory.
    agent_debug_log("inductor_segment.cc:plan", "before_gt_positions", "H2", "{}");
    graph::GraphTensor gt_positions(positions);
    agent_debug_log("inductor_segment.cc:plan", "after_gt_positions", "H2", "{}");
    agent_debug_log("inductor_segment.cc:plan", "before_gt_hidden", "H2", "{}");
    graph::GraphTensor gt_hidden(hidden_states);
    agent_debug_log("inductor_segment.cc:plan", "after_gt_hidden", "H2", "{}");
    agent_debug_log("inductor_segment.cc:plan", "before_gt_residual", "H2", "{}");
    graph::GraphTensor gt_residual(residual);
    agent_debug_log("inductor_segment.cc:plan", "after_gt_residual", "H2", "{}");
    agent_debug_log("inductor_segment.cc:plan", "before_gt_qkv", "H2", "{}");
    graph::GraphTensor gt_q(q_rope);
    graph::GraphTensor gt_k(k_rope);
    graph::GraphTensor gt_v(v_rope);
    agent_debug_log("inductor_segment.cc:plan", "after_gt_qkv", "H2", "{}");
    auto *meta = new PlannedMeta{
        std::move(gt_positions),
        positions_padded,
        std::move(gt_hidden),
        std::move(gt_residual),
        std::move(gt_q),
        std::move(gt_k),
        std::move(gt_v),
        segment_id,
        layer_idx,
        bucket,
        valid_len,
    };
    agent_debug_log("inductor_segment.cc:plan", "plan_done", "H5", "{}");
    return meta;
}

void run(void *planned_meta) {
#ifdef ENABLE_ATEN
    auto *meta = reinterpret_cast<PlannedMeta *>(planned_meta);
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
    const size_t valid_len = resolve_valid_seq_len(meta->bucket, hidden_states);
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
        valid_len);
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

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(InductorSegment, &plan, &run, &cleanup);

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

} // namespace inductor_segment_impl

} // namespace infinicore::op
