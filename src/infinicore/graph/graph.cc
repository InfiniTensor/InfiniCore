#include "graph_manager.hpp"

#include "../../bridge/infini/rt.hpp"
#include "../utils.hpp"
#include "infinicore/context/context.hpp"

#ifdef USE_INFINIRT_GRAPH
#include <cstdlib>
#include <infini/rt.h>
#endif

namespace infinicore::graph {

#ifdef USE_INFINIRT_GRAPH
namespace rt_runtime = ::infini::rt::runtime;
#endif

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
    rt_runtime::Graph graph = nullptr;
    rt_runtime::GraphExec exec = nullptr;
    rt_runtime::Stream stream = nullptr;
    ::infini::rt::Device::Type device_type = ::infini::rt::Device::Type::kCount;
    int device_index = 0;

    ~DeviceGraph() {
        if (exec) {
            (void)rt_runtime::GraphExecDestroy(exec);
        }
        if (graph) {
            (void)rt_runtime::GraphDestroy(graph);
        }
    }

    void launch() {
        ::infini::rt::set_runtime_device_type(device_type);
        INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(rt_runtime::SetDevice(device_index)));
        INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(rt_runtime::GraphLaunch(exec, stream)));
    }
};

struct Graph::Segment {
    bool capture_safe;
    std::vector<std::shared_ptr<GraphOperator>> ops;
    std::unique_ptr<DeviceGraph> device_graph;

    explicit Segment(bool capture_safe_) : capture_safe(capture_safe_) {
    }

    void run() const {
        if (device_graph) {
            device_graph->launch();
            return;
        }
        for (const auto &op : ops) {
            op->run();
        }
    }
};
#else
struct Graph::DeviceGraph {};
struct Graph::Segment {};
#endif

Graph::Graph() {
}

void Graph::run() const {
#ifdef USE_INFINIRT_GRAPH
    if (segments_.empty()) {
        for (const auto &op : op_list_) {
            op->run();
        }
        return;
    }
    for (const auto &segment : segments_) {
        segment->run();
    }
    return;
#endif
    for (const auto &op : op_list_) {
        op->run();
    }
}

void Graph::add_operator(std::shared_ptr<GraphOperator> op) {
    op_list_.push_back(op);
}

void Graph::instantiate() {
#ifdef USE_INFINIRT_GRAPH
    segments_.clear();
    auto current_device = context::getDevice();
    auto device_type = bridge::infini::rt::translate_to(static_cast<infiniDevice_t>(current_device.getType()));
    auto device_index = static_cast<int>(current_device.getIndex());
    auto stream = bridge::infini::rt::translate_to(context::getStream());
    if (device_type == ::infini::rt::Device::Type::kCount) {
        spdlog::warn("InfiniRT graph runtime does not support the current device. Falling back to eager execution.");
        return;
    }
    ::infini::rt::set_runtime_device_type(device_type);
    auto set_device_status = bridge::infini::rt::translate(rt_runtime::SetDevice(device_index));
    if (set_device_status != INFINI_STATUS_SUCCESS) {
        spdlog::warn("InfiniRT graph runtime failed to select the current device. Falling back to eager execution.");
        return;
    }

    // Warm the complete op list before splitting it into replay segments.
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

    for (const auto &op : op_list_) {
        const bool capture_safe = op->is_device_graph_capture_safe();
        if (segments_.empty() || segments_.back()->capture_safe != capture_safe) {
            segments_.push_back(std::make_unique<Segment>(capture_safe));
        }
        segments_.back()->ops.push_back(op);
    }

    for (auto &segment : segments_) {
        if (!segment->capture_safe) {
            // Replay non-capturable operators once between captured segments so
            // later capture observes the same stream-ordered dependencies.
            segment->run();
            continue;
        }

        segment->device_graph = std::make_unique<DeviceGraph>();
        auto &device_graph = *segment->device_graph;
        device_graph.device_type = device_type;
        device_graph.device_index = device_index;
        device_graph.stream = stream;

        auto begin_status = bridge::infini::rt::translate(rt_runtime::StreamBeginCapture(
            device_graph.stream,
            rt_runtime::StreamCaptureMode::kStreamCaptureModeRelaxed));
        if (begin_status != INFINI_STATUS_SUCCESS) {
            throw std::runtime_error("failed to begin device graph capture");
        }

        for (const auto &op : segment->ops) {
            op->run();
        }

        auto end_status = bridge::infini::rt::translate(rt_runtime::StreamEndCapture(
            device_graph.stream,
            &device_graph.graph));
        if (end_status != INFINI_STATUS_SUCCESS) {
            throw std::runtime_error("failed to end device graph capture");
        }

        auto instantiate_status = bridge::infini::rt::translate(rt_runtime::GraphInstantiate(
            &device_graph.exec,
            device_graph.graph));
        if (instantiate_status != INFINI_STATUS_SUCCESS) {
            throw std::runtime_error("failed to instantiate device graph");
        }
    }

    static bool logged_once = false;
    if (!logged_once) {
        logged_once = true;
        spdlog::info("Using InfiniRT C++ graph runtime API for graph capture and replay.");
    }

    if (std::getenv("INFINICORE_GRAPH_DEBUG") != nullptr) {
        size_t host_segments = 0;
        for (const auto &segment : segments_) {
            host_segments += segment->capture_safe ? 0 : 1;
        }
        spdlog::info(
            "segmented graph: operators={}, segments={}, host_segments={}",
            op_list_.size(), segments_.size(), host_segments);
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

void GraphManager::cancel_recording() {
    recording_ = false;
    graph_.reset();
}

} // namespace infinicore::graph
