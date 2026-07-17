#include "graph_manager.hpp"

#include "../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/graph/capture_arena.hpp"
#include <cstdlib>
#include <cstring>
#include <infinirt.h>
#include <string>
#include <utility>

#if defined(ENABLE_ATEN) \
    && (defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API))
#include <c10/cuda/CUDAGuard.h>
#include "infinicore/adaptor/aten_adaptor.hpp"
#define INFINI_TORCH_CAPTURE_STREAM_ALIGN 1
#endif

namespace infinicore::graph {

namespace {

bool graph_strict_replay_enabled() {
    const char *v = std::getenv("INFINI_GRAPH_STRICT_REPLAY");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

infinirtStreamCaptureMode_t graph_capture_mode() {
    const char *v = std::getenv("INFINI_GRAPH_CAPTURE_MODE");
    if (v == nullptr) {
        return INFINIRT_STREAM_CAPTURE_MODE_RELAXED;
    }
    if (std::strcmp(v, "global") == 0) {
        return INFINIRT_STREAM_CAPTURE_MODE_GLOBAL;
    }
    if (std::strcmp(v, "thread_local") == 0) {
        return INFINIRT_STREAM_CAPTURE_MODE_THREAD_LOCAL;
    }
    return INFINIRT_STREAM_CAPTURE_MODE_RELAXED;
}

[[noreturn]] void throw_graph_fatal(const std::string &msg) {
    throw std::runtime_error(msg);
}

} // namespace

/* =========================
 * GraphTensor
 * ========================= */

GraphTensor::GraphTensor(const Tensor &tensor) : Tensor(tensor->to_blob_()) {
}

/* =========================
 * GraphOperator
 * ========================= */

void DispatchableGraphOperator::run() const {
    runner_(planned_meta_);
}

DispatchableGraphOperator::~DispatchableGraphOperator() {
    if (deleter_) {
        deleter_(&planned_meta_);
    }
}

/* =========================
 * Graph
 * ========================= */

struct Graph::DeviceGraph {
    infinirtGraph_t graph{nullptr};
    infinirtGraphExec_t exec{nullptr};
    infinirtGraphNode_t node{nullptr};
    std::vector<char> log_buffer;
    /// Capture-lifetime MoE/ATen temps (IC-owned + residual torch retain).
    std::unique_ptr<CaptureArena> capture_arena;

    DeviceGraph() {
        log_buffer.resize(4 * 1024);
    }

    ~DeviceGraph() {
        if (exec) {
            infinirtGraphExecDestroy(exec);
        }
        if (graph) {
            infinirtGraphDestroy(graph);
        }
        // capture_arena destroyed after graph exec (temps must outlive replay).
    }

    void launch() {
        if (infinirtGraphLuanch(exec, context::getStream()) != INFINI_STATUS_SUCCESS) {
            throw std::runtime_error("infinirtGraphLuanch failed");
        }
    }
};

void Graph::run_ops_(const std::vector<std::shared_ptr<GraphOperator>> &ops) const {
    for (auto &op : ops) {
        op->run();
    }
}

void Graph::run_op_list_() const {
    run_ops_(op_list_);
}

Graph::Graph() {
}

void Graph::run() const {
    if (replay_steps_.empty()) {
        run_op_list_();
        last_replay_used_device_ = false;
        return;
    }

    bool used_device = false;
    for (const auto &step : replay_steps_) {
        if (step.kind == ReplayStep::Kind::HostOp) {
            run_ops_(step.ops);
            continue;
        }
        if (step.device != nullptr && step.device->exec != nullptr) {
            if (infinirtGraphLuanch(step.device->exec, context::getStream()) == INFINI_STATUS_SUCCESS) {
                ++replay_device_ok_;
                used_device = true;
                continue;
            }
            ++replay_op_list_fallback_;
            spdlog::warn("hcGraphLaunch replay failed; falling back to op-list segment");
            if (graph_strict_replay_enabled()) {
                throw std::runtime_error("hcGraphLaunch replay failed (INFINI_GRAPH_STRICT_REPLAY=1)");
            }
            run_ops_(step.ops);
            continue;
        }
        run_ops_(step.ops);
    }
    last_replay_used_device_ = used_device;
}

bool Graph::has_device_exec() const {
    return device_segment_count() > 0;
}

size_t Graph::device_segment_count() const {
    size_t n = 0;
    for (const auto &step : replay_steps_) {
        if (step.kind == ReplayStep::Kind::DeviceSegment && step.device != nullptr
            && step.device->exec != nullptr) {
            ++n;
        }
    }
    return n;
}

std::string Graph::device_graph_log() const {
    std::string out;
    for (const auto &step : replay_steps_) {
        if (step.device == nullptr) {
            continue;
        }
        const auto &buf = step.device->log_buffer;
        const size_t len = strnlen(buf.data(), buf.size());
        if (len == 0) {
            continue;
        }
        if (!out.empty()) {
            out.push_back('\n');
        }
        out.append(buf.data(), len);
    }
    return out;
}

bool Graph::last_replay_used_device() const {
    return last_replay_used_device_;
}

uint64_t Graph::replay_device_ok() const {
    return replay_device_ok_;
}

uint64_t Graph::replay_op_list_fallback() const {
    return replay_op_list_fallback_;
}

size_t Graph::capture_arena_bytes() const {
    size_t n = 0;
    for (const auto &step : replay_steps_) {
        if (step.device != nullptr && step.device->capture_arena) {
            n += step.device->capture_arena->bytes_allocated();
        }
    }
    return n;
}

size_t Graph::capture_arena_blocks() const {
    size_t n = 0;
    for (const auto &step : replay_steps_) {
        if (step.device != nullptr && step.device->capture_arena) {
            n += step.device->capture_arena->num_blocks();
        }
    }
    return n;
}

size_t Graph::capture_arena_retained_torch() const {
    size_t n = 0;
    for (const auto &step : replay_steps_) {
        if (step.device != nullptr && step.device->capture_arena) {
            n += step.device->capture_arena->num_retained_torch();
        }
    }
    return n;
}

void Graph::add_operator(std::shared_ptr<GraphOperator> op) {
    op_list_.push_back(op);
}

void Graph::capture_device_segment_(ReplayStep &step) {
    step.device = std::make_unique<DeviceGraph>();
    step.device->capture_arena = std::make_unique<CaptureArena>();

#ifdef INFINI_TORCH_CAPTURE_STREAM_ALIGN
    // Align Torch current stream with InfiniCore capture stream to avoid
    // "legacy stream depend on a capturing blocking stream". Memory ownership
    // is InfiniCore CaptureArena (no c10::cuda::MemPool).
    const c10::cuda::CUDAStream torch_stream = infinicore::adaptor::get_cuda_stream();
    c10::cuda::CUDAStreamGuard stream_guard(torch_stream);
#endif

    begin_capture_arena(*step.device->capture_arena);
    struct ArenaGuard {
        CaptureArena *arena;
        bool active{true};
        ~ArenaGuard() {
            if (active && arena != nullptr) {
                end_capture_arena(*arena);
            }
        }
        void release() {
            if (active && arena != nullptr) {
                end_capture_arena(*arena);
                active = false;
            }
        }
    } arena_guard{step.device->capture_arena.get()};

    const auto capture_mode = graph_capture_mode();
    if (infinirtStreamBeginCapture(context::getStream(), capture_mode)
        != INFINI_STATUS_SUCCESS) {
        arena_guard.release();
        if (graph_strict_replay_enabled()) {
            throw_graph_fatal("infinirtStreamBeginCapture failed (INFINI_GRAPH_STRICT_REPLAY=1)");
        }
        return;
    }

    context::setDeviceStreamCapturing(true);
    try {
        // Host-break ops are never placed in DeviceSegment steps.
        run_ops_(step.ops);
    } catch (...) {
        context::setDeviceStreamCapturing(false);
        arena_guard.release();
        (void)infinirtStreamEndCapture(context::getStream(), &step.device->graph);
        throw;
    }
    context::setDeviceStreamCapturing(false);

    if (infinirtStreamEndCapture(context::getStream(), &step.device->graph)
        != INFINI_STATUS_SUCCESS) {
        arena_guard.release();
        if (graph_strict_replay_enabled()) {
            throw_graph_fatal("infinirtStreamEndCapture failed (INFINI_GRAPH_STRICT_REPLAY=1)");
        }
        return;
    }
    // Keep arena allocations for DeviceGraph lifetime; only drop TLS / pin mode.
    arena_guard.release();

    if (infinirtGraphInstantiate(
            &step.device->exec,
            step.device->graph,
            &step.device->node,
            step.device->log_buffer.data(),
            step.device->log_buffer.size())
        != INFINI_STATUS_SUCCESS) {
        const size_t len = strnlen(step.device->log_buffer.data(), step.device->log_buffer.size());
        const std::string log_msg(step.device->log_buffer.data(), len);
        if (graph_strict_replay_enabled()) {
            throw_graph_fatal(
                "infinirtGraphInstantiate failed (INFINI_GRAPH_STRICT_REPLAY=1): " + log_msg);
        }
        spdlog::warn("Fail to instantiate device graph segment (op-list fallback): {}", log_msg);
        return;
    }

    const bool probe_ok = step.device->exec == nullptr
                          || infinirtGraphLuanch(step.device->exec, context::getStream())
                                 == INFINI_STATUS_SUCCESS;
    if (step.device->exec != nullptr && !probe_ok) {
        if (graph_strict_replay_enabled()) {
            infinirtGraphExecDestroy(step.device->exec);
            step.device->exec = nullptr;
            infinicore::context::syncStream();
            throw_graph_fatal(
                "hcGraphLaunch instantiate probe failed (INFINI_GRAPH_STRICT_REPLAY=1)");
        }
        spdlog::warn("Device graph segment launch probe failed; op-list replay fallback");
        infinirtGraphExecDestroy(step.device->exec);
        step.device->exec = nullptr;
        infinicore::context::syncStream();
    }
}

void Graph::instantiate() {
    replay_steps_.clear();

    // Pre-capture warmup: full op-list (including host-break MoE) outside capture.
    for (size_t iter = 0; iter < 5; ++iter) {
        run_op_list_();
    }
    infinicore::context::syncStream();

#ifdef USE_INFINIRT_GRAPH
    // Split around host-break ops so Triton MoE never enters stream capture.
    ReplayStep pending;
    pending.kind = ReplayStep::Kind::DeviceSegment;
    auto flush_device = [this](ReplayStep &seg) {
        if (seg.ops.empty()) {
            return;
        }
        capture_device_segment_(seg);
        replay_steps_.push_back(std::move(seg));
        seg = ReplayStep{};
        seg.kind = ReplayStep::Kind::DeviceSegment;
    };

    for (auto &op : op_list_) {
        if (op->is_host_break()) {
            flush_device(pending);
            ReplayStep host;
            host.kind = ReplayStep::Kind::HostOp;
            host.ops.push_back(op);
            replay_steps_.push_back(std::move(host));
            continue;
        }
        pending.ops.push_back(op);
    }
    flush_device(pending);
#else
    (void)0;
#endif
}

Graph::~Graph() = default;

/* =========================
 * GraphManager
 * ========================= */

bool GraphManager::is_recording() const {
    return recording_;
}

void GraphManager::start_recording() {
    if (is_recording()) {
        spdlog::warn("Graph is already recording. Previous recording will be dropped.");
    }
    recording_ = true;
    graph_ = std::make_shared<Graph>();
}

void GraphManager::add_operator(std::shared_ptr<GraphOperator> op) {
    INFINICORE_ASSERT(is_recording());

    graph_->add_operator(op);
}

std::shared_ptr<Graph> GraphManager::stop_recording() {
    if (!is_recording()) {
        spdlog::warn("Graph is not recording. Please start recording first.");
        return nullptr;
    }
    recording_ = false;
#ifdef USE_INFINIRT_GRAPH
    graph_->instantiate();
#endif
    return std::exchange(graph_, nullptr);
}

} // namespace infinicore::graph
