#include "graph_manager.hpp"

#include "../utils.hpp"
#include "infinicore/context/context.hpp"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <chrono>
#include <infinirt.h>
#include <string>

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

// #region agent log
void agent_debug_log(
    const char *location,
    const char *message,
    const char *hypothesis_id,
    const std::string &data_json) {
    std::ofstream out(
        "/opt/offline/infinilm-metax-20260622/.cursor/debug-1c7a11.log",
        std::ios::app);
    if (!out) {
        return;
    }
    const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
    out << "{\"sessionId\":\"1c7a11\",\"location\":\"" << location
        << "\",\"message\":\"" << message << "\",\"hypothesisId\":\"" << hypothesis_id
        << "\",\"data\":" << data_json << ",\"timestamp\":" << ts << "}\n";
}
// #endregion

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
    infinirtGraph_t graph;
    infinirtGraphExec_t exec;
    infinirtGraphNode_t node;
    std::vector<char> log_buffer;

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
    }

    void launch() {
        if (infinirtGraphLuanch(exec, context::getStream()) != INFINI_STATUS_SUCCESS) {
            throw std::runtime_error("infinirtGraphLuanch failed");
        }
    }
};

void Graph::run_op_list_() const {
    for (auto &op : op_list_) {
        op->run();
    }
}

Graph::Graph() {
}

void Graph::run() const {
    if (device_graph_ != nullptr && device_graph_->exec != nullptr) {
        // #region agent log
        agent_debug_log(
            "graph.cc:Graph::run",
            "device_exec_launch_attempt",
            "H1",
            std::string("{\"stream_ptr\":") + std::to_string(
                reinterpret_cast<uintptr_t>(context::getStream()))
                + ",\"exec_ptr\":"
                + std::to_string(reinterpret_cast<uintptr_t>(device_graph_->exec)) + "}");
        // #endregion
        if (infinirtGraphLuanch(device_graph_->exec, context::getStream()) == INFINI_STATUS_SUCCESS) {
            ++replay_device_ok_;
            last_replay_used_device_ = true;
            // #region agent log
            agent_debug_log(
                "graph.cc:Graph::run",
                "device_exec_launch_ok",
                "H1",
                "{\"ok\":true}");
            // #endregion
            return;
        }
        ++replay_op_list_fallback_;
        last_replay_used_device_ = false;
        // #region agent log
        agent_debug_log(
            "graph.cc:Graph::run",
            "device_exec_launch_failed",
            "H1",
            "{\"ok\":false,\"strict\":"
                + std::string(graph_strict_replay_enabled() ? "true" : "false") + "}");
        // #endregion
        spdlog::warn("hcGraphLaunch replay failed; falling back to op-list replay");
        if (graph_strict_replay_enabled()) {
            throw std::runtime_error("hcGraphLaunch replay failed (INFINI_GRAPH_STRICT_REPLAY=1)");
        }
        run_op_list_();
        return;
    }
    run_op_list_();
}

bool Graph::has_device_exec() const {
    return device_graph_ != nullptr && device_graph_->exec != nullptr;
}

std::string Graph::device_graph_log() const {
    if (device_graph_ == nullptr) {
        return {};
    }
    const auto &buf = device_graph_->log_buffer;
    const size_t len = strnlen(buf.data(), buf.size());
    return std::string(buf.data(), len);
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

void Graph::add_operator(std::shared_ptr<GraphOperator> op) {
    op_list_.push_back(op);
}

void Graph::instantiate() {
    // Reset device graph
    device_graph_ = std::make_unique<DeviceGraph>();

    // Pre-capture warmup: op-list only (device exec is null; avoid probing stale exec).
    for (size_t iter = 0; iter < 5; ++iter) {
        run_op_list_();
    }
    infinicore::context::syncStream();

    const auto capture_mode = graph_capture_mode();
    // #region agent log
    agent_debug_log(
        "graph.cc:Graph::instantiate",
        "begin_capture",
        "H5",
        std::string("{\"capture_mode\":") + std::to_string(static_cast<int>(capture_mode))
            + ",\"op_count\":" + std::to_string(op_list_.size()) + "}");
    // #endregion
    if (infinirtStreamBeginCapture(context::getStream(), capture_mode)
        != INFINI_STATUS_SUCCESS) {
        if (graph_strict_replay_enabled()) {
            throw_graph_fatal("infinirtStreamBeginCapture failed (INFINI_GRAPH_STRICT_REPLAY=1)");
        }
        return;
    }

    // Record op-list into hcGraph (exec still null during capture).
    run_op_list_();

    if (infinirtStreamEndCapture(
            context::getStream(),
            &device_graph_.get()->graph)
        != INFINI_STATUS_SUCCESS) {
        if (graph_strict_replay_enabled()) {
            throw_graph_fatal("infinirtStreamEndCapture failed (INFINI_GRAPH_STRICT_REPLAY=1)");
        }
        return;
    }

    if (infinirtGraphInstantiate(
            &device_graph_.get()->exec,
            device_graph_.get()->graph,
            &device_graph_.get()->node,
            device_graph_.get()->log_buffer.data(),
            device_graph_.get()->log_buffer.size())
        != INFINI_STATUS_SUCCESS) {
        const std::string log_msg = device_graph_log();
        // #region agent log
        agent_debug_log(
            "graph.cc:Graph::instantiate",
            "instantiate_failed",
            "H5",
            "{\"ok\":false}");
        // #endregion
        if (graph_strict_replay_enabled()) {
            throw_graph_fatal(
                "infinirtGraphInstantiate failed (INFINI_GRAPH_STRICT_REPLAY=1): " + log_msg);
        }
        spdlog::warn("Fail to instantiate device graph (op-list replay fallback): {}", log_msg);
    } else {
        const bool probe_ok = device_graph_->exec == nullptr
                              || infinirtGraphLuanch(device_graph_->exec, context::getStream())
                                     == INFINI_STATUS_SUCCESS;
        // #region agent log
        agent_debug_log(
            "graph.cc:Graph::instantiate",
            "instantiate_done",
            "H5",
            std::string("{\"exec_ptr\":")
                + std::to_string(reinterpret_cast<uintptr_t>(device_graph_->exec))
                + ",\"probe_ok\":" + (probe_ok ? "true" : "false") + "}");
        // #endregion
        if (device_graph_->exec != nullptr && !probe_ok) {
            if (graph_strict_replay_enabled()) {
                infinirtGraphExecDestroy(device_graph_->exec);
                device_graph_->exec = nullptr;
                infinicore::context::syncStream();
                throw_graph_fatal(
                    "hcGraphLaunch instantiate probe failed (INFINI_GRAPH_STRICT_REPLAY=1)");
            }
            spdlog::warn("Device graph exec launch probe failed; op-list replay fallback");
            infinirtGraphExecDestroy(device_graph_->exec);
            device_graph_->exec = nullptr;
            infinicore::context::syncStream();
        }
    }
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
