#include "graph_manager.hpp"

#include "../utils.hpp"
#include "infinicore/context/context.hpp"
#include <cstdlib>
#include <cstring>
#include <infinirt.h>
#include <string>

namespace infinicore::graph {

namespace {

bool graph_strict_replay_enabled() {
    const char *v = std::getenv("INFINI_GRAPH_STRICT_REPLAY");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
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
        if (infinirtGraphLuanch(device_graph_->exec, context::getStream()) == INFINI_STATUS_SUCCESS) {
            ++replay_device_ok_;
            last_replay_used_device_ = true;
            return;
        }
        ++replay_op_list_fallback_;
        last_replay_used_device_ = false;
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

    // warmup
    for (size_t iter = 0; iter < 5; ++iter) {
        this->run();
    }
    infinicore::context::syncStream();

    if (infinirtStreamBeginCapture(
            context::getStream(),
            INFINIRT_STREAM_CAPTURE_MODE_RELAXED)
        != INFINI_STATUS_SUCCESS) {
        return;
    }

    // Run and record
    this->run();

    if (infinirtStreamEndCapture(
            context::getStream(),
            &device_graph_.get()->graph)
        != INFINI_STATUS_SUCCESS) {
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
        spdlog::warn("Fail to instantiate device graph (op-list replay fallback): {}", log_msg);
    } else if (device_graph_->exec != nullptr
               && infinirtGraphLuanch(device_graph_->exec, context::getStream()) != INFINI_STATUS_SUCCESS) {
        spdlog::warn("Device graph exec launch probe failed; op-list replay fallback");
        infinirtGraphExecDestroy(device_graph_->exec);
        device_graph_->exec = nullptr;
        infinicore::context::syncStream();
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
