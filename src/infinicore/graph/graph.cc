#include "graph_manager.hpp"

#include "../utils.hpp"
#include "infinicore/context/context.hpp"

#ifdef USE_INFINIRT_GRAPH
#include "standalone_infinirt_graph_bridge.hpp"
#include <infinirt.h>
#endif

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

#ifdef USE_INFINIRT_GRAPH
struct Graph::DeviceGraph {
    infinirtGraph_t graph = nullptr;
    infinirtGraphExec_t exec = nullptr;
    infinirtGraphNode_t node = nullptr;
    infinirtStream_t stream = nullptr;
    bool standalone = false;
    std::vector<char> log_buffer;

    DeviceGraph() {
        log_buffer.resize(4 * 1024);
    }

    ~DeviceGraph() {
        if (exec) {
            if (standalone) {
                standalone_infinirt::graph_exec_destroy(exec);
            } else {
                infinirtGraphExecDestroy(exec);
            }
        }
        if (graph) {
            if (standalone) {
                standalone_infinirt::graph_destroy(graph);
            } else {
                infinirtGraphDestroy(graph);
            }
        }
    }

    void launch() {
        if (standalone) {
            INFINICORE_CHECK_ERROR(standalone_infinirt::graph_launch(exec, stream));
        } else {
            INFINICORE_CHECK_ERROR(infinirtGraphLuanch(exec, context::getStream()));
        }
    }
};
#else
struct Graph::DeviceGraph {};
#endif

Graph::Graph() {
}

void Graph::run() const {
#ifdef USE_INFINIRT_GRAPH
    if (device_graph_ != nullptr && device_graph_.get()->exec != nullptr) {
        device_graph_.get()->launch();
        return;
    }
#endif
    for (auto &op : op_list_) {
        op->run();
    }
}

void Graph::add_operator(std::shared_ptr<GraphOperator> op) {
    op_list_.push_back(op);
}

void Graph::instantiate() {
#ifdef USE_INFINIRT_GRAPH
    // Reset device graph
    device_graph_ = std::make_unique<DeviceGraph>();
    device_graph_->standalone = standalone_infinirt::available(context::getDevice());
    device_graph_->stream = context::getStream();
    if (device_graph_->standalone) {
        auto set_device_status = standalone_infinirt::set_device(context::getDevice());
        if (set_device_status != INFINI_STATUS_SUCCESS) {
            spdlog::warn("Standalone InfiniRT graph bridge failed to select the current device. Falling back to eager execution.");
            device_graph_.reset();
            return;
        }

        static bool logged_once = false;
        if (!logged_once) {
            logged_once = true;
            spdlog::info("Using standalone InfiniRT C++ graph runtime API for graph capture and replay.");
        }
    }

    // warmup
    for (size_t iter = 0; iter < 5; ++iter) {
        this->run();
    }
    infinicore::context::syncStream();

    auto begin_status = device_graph_->standalone
                          ? standalone_infinirt::stream_begin_capture(device_graph_->stream, INFINIRT_STREAM_CAPTURE_MODE_RELAXED)
                          : infinirtStreamBeginCapture(context::getStream(), INFINIRT_STREAM_CAPTURE_MODE_RELAXED);
    if (begin_status != INFINI_STATUS_SUCCESS) {
        spdlog::warn("Fail to begin device graph capture.");
        device_graph_.reset();
        return;
    }

    // Run and record
    this->run();

    auto end_status = device_graph_->standalone
                        ? standalone_infinirt::stream_end_capture(device_graph_->stream, &device_graph_.get()->graph)
                        : infinirtStreamEndCapture(context::getStream(), &device_graph_.get()->graph);
    if (end_status != INFINI_STATUS_SUCCESS) {
        spdlog::warn("Fail to end device graph capture.");
        device_graph_.reset();
        return;
    }

    auto instantiate_status = device_graph_->standalone
                                ? standalone_infinirt::graph_instantiate(&device_graph_.get()->exec, device_graph_.get()->graph)
                                : infinirtGraphInstantiate(
                                    &device_graph_.get()->exec,
                                    device_graph_.get()->graph,
                                    &device_graph_.get()->node,
                                    device_graph_.get()->log_buffer.data(),
                                    device_graph_.get()->log_buffer.size());
    if (instantiate_status != INFINI_STATUS_SUCCESS) {
        static bool warned_once = false;
        if (!warned_once) {
            warned_once = true;
            spdlog::warn("Fail to instantiate device graph: {}", std::string(device_graph_.get()->log_buffer.data()));
        }
        device_graph_.reset();
    }
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
