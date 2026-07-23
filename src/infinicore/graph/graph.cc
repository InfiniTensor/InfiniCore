#include "graph_manager.hpp"

#include "../utils.hpp"
#include "infinicore/context/context.hpp"
#include <cstdlib>
#include <infinirt.h>

namespace infinicore::graph {

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
        graph = nullptr;
        exec = nullptr;
        node = nullptr;
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
        INFINICORE_CHECK_ERROR(infinirtGraphLuanch(exec, context::getStream()));
    }
};

struct Graph::ReplayStep {
    std::unique_ptr<DeviceGraph> device_graph;
    std::shared_ptr<GraphOperator> host_op;
};

Graph::Graph() {
}

void Graph::run() const {
    if (replay_steps_.empty()) {
        for (auto &op : op_list_) {
            op->run();
            if (op->requires_stream_sync_after_run()) {
                infinicore::context::syncStream();
            }
        }
        return;
    }
    for (const auto &step : replay_steps_) {
        if (step->device_graph) {
            step->device_graph->launch();
        } else {
            step->host_op->run();
            if (step->host_op->requires_stream_sync_after_run()) {
                infinicore::context::syncStream();
            }
        }
    }
}

void Graph::add_operator(std::shared_ptr<GraphOperator> op) {
    op_list_.push_back(op);
}

void Graph::instantiate() {
    replay_steps_.clear();

    // Warm the complete op list before splitting it at host-replayed P2P ops.
    for (size_t iter = 0; iter < 5; ++iter) {
        this->run();
    }
    infinicore::context::syncStream();

    // Diagnostic escape hatch: keep GraphTensor/operator replay semantics but
    // bypass device-graph capture, including segmented PP capture.
    if (std::getenv("INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS") != nullptr) {
        spdlog::info("device graph segments disabled; replaying recorded operators");
        return;
    }

    auto capture_segment = [&](size_t begin, size_t end) {
        if (begin == end) {
            return;
        }
        auto device_graph = std::make_unique<DeviceGraph>();
        if (infinirtStreamBeginCapture(
                context::getStream(),
                INFINIRT_STREAM_CAPTURE_MODE_RELAXED)
            != INFINI_STATUS_SUCCESS) {
            throw std::runtime_error("failed to begin device graph capture");
        }

        for (size_t i = begin; i < end; ++i) {
            op_list_[i]->run();
        }

        if (infinirtStreamEndCapture(
                context::getStream(),
                &device_graph->graph)
            != INFINI_STATUS_SUCCESS) {
            throw std::runtime_error("failed to end device graph capture");
        }

        if (infinirtGraphInstantiate(
                &device_graph->exec,
                device_graph->graph,
                &device_graph->node,
                device_graph->log_buffer.data(),
                device_graph->log_buffer.size())
            != INFINI_STATUS_SUCCESS) {
            throw std::runtime_error(
                "failed to instantiate device graph: "
                + std::string(device_graph->log_buffer.data()));
        }
        auto step = std::make_unique<ReplayStep>();
        step->device_graph = std::move(device_graph);
        replay_steps_.push_back(std::move(step));
    };

    size_t segment_begin = 0;
    for (size_t i = 0; i < op_list_.size(); ++i) {
        if (op_list_[i]->is_device_graph_capture_safe()) {
            continue;
        }
        capture_segment(segment_begin, i);
        // Execute the P2P operation once between captured compute segments so
        // downstream capture sees the same stream-ordered data dependency.
        op_list_[i]->run();
        if (op_list_[i]->requires_stream_sync_after_run()) {
            infinicore::context::syncStream();
        }
        auto step = std::make_unique<ReplayStep>();
        step->host_op = op_list_[i];
        replay_steps_.push_back(std::move(step));
        segment_begin = i + 1;
    }
    capture_segment(segment_begin, op_list_.size());
    if (std::getenv("INFINICORE_GRAPH_DEBUG") != nullptr) {
        size_t host_steps = 0;
        for (const auto &step : replay_steps_) {
            host_steps += step->host_op != nullptr ? 1 : 0;
        }
        spdlog::info(
            "segmented graph: operators={}, replay_steps={}, host_steps={}",
            op_list_.size(), replay_steps_.size(), host_steps);
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

void GraphManager::cancel_recording() {
    recording_ = false;
    graph_.reset();
}

} // namespace infinicore::graph
