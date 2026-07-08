#include "infinicore/ops/inductor_segment.hpp"

#include "../../utils.hpp"
#include "inductor_segment_registry.hpp"

#ifdef ENABLE_ATEN
#include "aot_package_runner.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/context/context.hpp"
#endif

#include <fstream>
#include <chrono>
#include <stdexcept>

namespace {

// #region agent log
void agent_debug_log(
    const char *location,
    const char *message,
    const char *hypothesis_id,
    const std::string &data_json) {
    std::ofstream out(
        "/opt/offline/infinilm-metax-20260622/.cursor/debug-9ddc7d.log",
        std::ios::app);
    if (!out) {
        return;
    }
    const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
    out << "{\"sessionId\":\"9ddc7d\",\"location\":\"" << location
        << "\",\"message\":\"" << message << "\",\"hypothesisId\":\"" << hypothesis_id
        << "\",\"data\":" << data_json << ",\"timestamp\":" << ts << "}\n";
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
}

void copy_tensor_if_needed(infinicore::Tensor &dst, const at::Tensor &src) {
    auto dst_aten = infinicore::adaptor::to_aten_tensor(dst);
    if (dst_aten.data_ptr() == src.data_ptr()) {
        return;
    }
    if (src.is_contiguous()) {
        dst_aten.copy_(src);
    } else {
        dst_aten.copy_(src.contiguous());
    }
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
    auto &runner = infinicore::op::inductor_segment_impl::InductorSegmentRegistry::instance().runner(
        infinicore::op::PiecewiseInductorSegmentId::PreAttn, layer_idx, bucket);

    fill_positions_padded_into(positions_padded, positions, bucket, valid_len);

    auto hidden = infinicore::adaptor::to_aten_tensor(hidden_states);
    auto res = infinicore::adaptor::to_aten_tensor(residual);
    auto pos = infinicore::adaptor::to_aten_tensor(positions_padded);

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

    std::vector<at::Tensor> outputs = runner.run(
        {hidden, res, pos},
        infinicore::context::getStream());
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

namespace inductor_segment_impl {

struct PlannedMeta {
    graph::GraphTensor positions;
    graph::GraphTensor positions_padded;
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
    size_t valid_len = bucket;
    if (hidden_states->shape().size() >= 2) {
        valid_len = hidden_states->shape()[1];
    }
    const auto device = hidden_states->device();
    auto positions_padded = infinicore::Tensor::zeros(
        {1, bucket},
        positions->dtype(),
        device);
    return new PlannedMeta{
        graph::GraphTensor(positions),
        graph::GraphTensor(positions_padded),
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
}

void run(void *planned_meta) {
#ifdef ENABLE_ATEN
    auto *meta = reinterpret_cast<PlannedMeta *>(planned_meta);
    if (meta->segment_id != PiecewiseInductorSegmentId::PreAttn) {
        throw std::runtime_error(
            "InductorSegment: post_attn_cg AOT runtime not implemented yet");
    }
    Tensor positions_padded(meta->positions_padded);
    Tensor hidden_states(meta->hidden_states);
    Tensor residual(meta->residual);
    Tensor q_rope(meta->q_rope);
    Tensor k_rope(meta->k_rope);
    Tensor v_rope(meta->v_rope);
    run_pre_attn_segment(
        Tensor(meta->positions),
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
    auto &runner = InductorSegmentRegistry::instance().runner(
        PiecewiseInductorSegmentId::PreAttn, layer_idx, bucket);
    fill_positions_padded_into(positions_padded, positions, bucket, valid_len);
    auto hidden = infinicore::adaptor::to_aten_tensor(hidden_states);
    auto res = infinicore::adaptor::to_aten_tensor(residual);
    auto pos = infinicore::adaptor::to_aten_tensor(positions_padded);
    runner.warmup({hidden, res, pos});
}
#endif

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(InductorSegment, &plan, &run, &cleanup);

} // namespace inductor_segment_impl

} // namespace infinicore::op
