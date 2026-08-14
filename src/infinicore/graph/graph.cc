#include "graph_manager.hpp"

#include "../utils.hpp"
#include "infinicore/context/context.hpp"
#include <cstdlib>
#include <infinirt.h>

namespace infinicore::graph {

namespace {
using HostIntArrayMap = std::unordered_map<const void *, std::vector<int64_t>>;

thread_local const HostIntArrayMap *active_host_int_arrays = nullptr;

class HostIntArrayScope {
public:
    explicit HostIntArrayScope(const HostIntArrayMap *host_int_arrays)
        : previous_(active_host_int_arrays) {
        active_host_int_arrays = host_int_arrays;
    }

    ~HostIntArrayScope() {
        active_host_int_arrays = previous_;
    }

private:
    const HostIntArrayMap *previous_;
};
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

bool DispatchableGraphOperator::needs_graph_replay_update() const {
    return graph_replay_ != nullptr;
}

void DispatchableGraphOperator::graph_replay(GraphReplayStage stage) const {
    INFINICORE_ASSERT(graph_replay_ != nullptr);
    graph_replay_(planned_meta_, stage);
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
    std::vector<std::shared_ptr<GraphOperator>> updatable_ops;

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

    void launch() {
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
            for (const auto &op : device_graph->updatable_ops) {
                op->graph_replay(GraphReplayStage::UPDATE);
            }
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
    HostIntArrayScope host_int_array_scope(&host_int_arrays_);
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

void Graph::bind_host_int_array(const Tensor &device_tensor,
                                const int32_t *values,
                                size_t size) {
    INFINICORE_ASSERT(device_tensor);
    INFINICORE_ASSERT(values != nullptr || size == 0);

    auto &bound = host_int_arrays_[device_tensor->data()];
    bound.clear();
    bound.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        bound.push_back(values[i]);
    }
}

const std::vector<int64_t> *lookup_bound_host_int_array(const Tensor &tensor) {
    if (active_host_int_arrays == nullptr || !tensor) {
        return nullptr;
    }

    const auto it = active_host_int_arrays->find(tensor->data());
    if (it == active_host_int_arrays->end()) {
        return nullptr;
    }
    return &it->second;
}

void Graph::add_operator(std::shared_ptr<GraphOperator> op) {
    op_list_.push_back(op);
}

void Graph::instantiate() {
    segments_.clear();

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
        if (infinirtStreamBeginCapture(
                context::getStream(),
                INFINIRT_STREAM_CAPTURE_MODE_RELAXED)
            != INFINI_STATUS_SUCCESS) {
            throw std::runtime_error("failed to begin device graph capture");
        }

        for (const auto &op : segment->ops) {
            if (!op->needs_graph_replay_update()) {
                op->run();
                continue;
            }

            op->graph_replay(GraphReplayStage::CAPTURE_BEGIN);
            try {
                op->run();
            } catch (...) {
                try {
                    op->graph_replay(GraphReplayStage::CAPTURE_END);
                } catch (...) {
                    // Preserve the original capture failure.
                }
                throw;
            }
            op->graph_replay(GraphReplayStage::CAPTURE_END);
            device_graph.updatable_ops.push_back(op);
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
