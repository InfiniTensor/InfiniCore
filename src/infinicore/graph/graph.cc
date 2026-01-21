#include "graph_manager.hpp"

#include "../utils.hpp"
#include "infinicore/context/context.hpp"
#include <infinirt.h>

namespace infinicore::graph {

/* =========================
 * GraphTensor
 * ========================= */

GraphTensor::GraphTensor(const Tensor &tensor) : Tensor(tensor->to_blob()) {
}

void GraphTensor::resume() const {
    resume_from_blob_();
}

/* =========================
 * GraphOperator
 * ========================= */

void GraphOperator::run() const {
    runner_(planned_meta_);
}

GraphOperator::~GraphOperator() {
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
        INFINICORE_CHECK_ERROR(infinirtGraphLuanch(exec, context::getStream()));
    }
};

Graph::Graph() {
}

void Graph::run() const {
    if (device_graph_ != nullptr && device_graph_.get()->exec != nullptr) {
        device_graph_.get()->launch();
    } else {
        for (auto &op : op_list_) {
            op->run();
        }
    }
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
            INFINIRT_STREAM_CAPTURE_MODE_GLOBAL)
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
        static bool warned_once = false;
        if (!warned_once) {
            warned_once = true;
            spdlog::warn("Fail to instantiate device graph: {}", std::string(device_graph_.get()->log_buffer.data()));
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
    graph_->instantiate();
    return std::exchange(graph_, nullptr);
}

} // namespace infinicore::graph
