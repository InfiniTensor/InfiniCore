#include "graph_manager.hpp"

#include "../utils.hpp"
#include "infinicore/context/context.hpp"
#include <cstdlib>
#include <infinirt.h>

#include <algorithm>

#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <ATen/hip/HIPGraph.h>
#include <c10/hip/HIPGuard.h>

#include <mutex>
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

struct Graph::DeviceGraph {
    infinirtGraph_t graph = nullptr;
    infinirtGraphExec_t exec = nullptr;
    infinirtGraphNode_t node = nullptr;
    std::vector<char> log_buffer;
#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
    std::unique_ptr<at::cuda::CUDAGraph> hygon_graph;
#endif

    DeviceGraph() : graph(nullptr), exec(nullptr), node(nullptr) {
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

    bool is_instantiated() const {
#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
        if (hygon_graph) {
            return true;
        }
#endif
        return exec != nullptr;
    }

    void launch() {
#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
        if (hygon_graph) {
            c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
            hygon_graph->replay();
            return;
        }
#endif
        INFINICORE_CHECK_ERROR(infinirtGraphLuanch(exec, context::getStream()));
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

Graph::Graph() {
}

void Graph::run() const {
    if (segments_.empty()) {
        for (const auto &op : op_list_) {
            op->run();
        }
        return;
    }
    for (const auto &segment : segments_) {
        segment->run();
    }
}

void Graph::add_operator(std::shared_ptr<GraphOperator> op) {
    op_list_.push_back(op);
}

void Graph::instantiate() {
#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
    const bool is_hygon =
        context::getDevice().getType() == Device::Type::HYGON;
#else
    constexpr bool is_hygon = false;
#endif

    segments_.clear();

    // Warm the complete op list before splitting it into replay segments.
    for (size_t iter = 0; iter < 5; ++iter) {
        for (const auto &op : op_list_) {
            op->run();
        }
    }
    infinicore::context::syncStream();

    // Diagnostic escape hatch: keep GraphTensor/operator replay semantics but
    // bypass device-graph capture, including segmented PP capture.
    if (std::getenv("INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS") != nullptr) {
        spdlog::info("device graph segments disabled; replaying recorded operators");
        return;
    }

    for (const auto &op : op_list_) {
        const bool capture_safe = is_hygon || op->is_device_graph_capture_safe();
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
        if (infinirtStreamBeginCapture(
                context::getStream(),
                INFINIRT_STREAM_CAPTURE_MODE_RELAXED)
            != INFINI_STATUS_SUCCESS) {
            throw std::runtime_error("failed to begin device graph capture");
        }

        for (const auto &op : segment->ops) {
            op->run();
        }

        if (infinirtStreamEndCapture(
                context::getStream(),
                &device_graph.graph)
            != INFINI_STATUS_SUCCESS) {
            throw std::runtime_error("failed to end device graph capture");
        }

        if (infinirtGraphInstantiate(
                &device_graph.exec,
                device_graph.graph,
                &device_graph.node,
                device_graph.log_buffer.data(),
                device_graph.log_buffer.size())
            != INFINI_STATUS_SUCCESS) {
            throw std::runtime_error(
                "failed to instantiate device graph: "
                + std::string(device_graph.log_buffer.data()));
        }
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
