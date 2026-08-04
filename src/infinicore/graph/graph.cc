#include "graph_manager.hpp"

#include "../utils.hpp"
#include "infinicore/context/context.hpp"
#include <cstdlib>
#include <infinirt.h>

namespace infinicore::graph {
namespace {

using HostIntArrayMap = std::unordered_map<const void *, std::vector<int64_t>>;

thread_local const HostIntArrayMap *current_host_int_arrays = nullptr;
thread_local bool task_update_active = false;
thread_local HostIntArrayMap staged_host_int_arrays;
thread_local infinirtGraphTaskGroup_t *capture_task_group_handle = nullptr;
thread_local infinirtGraphTaskGroup_t task_update_handle = nullptr;

class HostIntArrayScope {
public:
    HostIntArrayScope(
        const HostIntArrayMap *host_int_arrays,
        bool is_task_update) {
        current_host_int_arrays = host_int_arrays;
        task_update_active = is_task_update;
    }

    ~HostIntArrayScope() {
        task_update_active = false;
        current_host_int_arrays = nullptr;
    }

    HostIntArrayScope(const HostIntArrayScope &) = delete;
    HostIntArrayScope &operator=(const HostIntArrayScope &) = delete;
};

class TaskGroupCaptureScope {
public:
    explicit TaskGroupCaptureScope(infinirtGraphTaskGroup_t *handle) {
        INFINICORE_ASSERT(capture_task_group_handle == nullptr);
        capture_task_group_handle = handle;
    }

    ~TaskGroupCaptureScope() {
        capture_task_group_handle = nullptr;
    }

    TaskGroupCaptureScope(const TaskGroupCaptureScope &) = delete;
    TaskGroupCaptureScope &operator=(const TaskGroupCaptureScope &) = delete;
};

class TaskUpdateHandleScope {
public:
    explicit TaskUpdateHandleScope(infinirtGraphTaskGroup_t handle) {
        INFINICORE_ASSERT(task_update_handle == nullptr);
        INFINICORE_ASSERT(handle != nullptr);
        task_update_handle = handle;
    }

    ~TaskUpdateHandleScope() {
        task_update_handle = nullptr;
    }

    TaskUpdateHandleScope(const TaskUpdateHandleScope &) = delete;
    TaskUpdateHandleScope &operator=(const TaskUpdateHandleScope &) = delete;
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

DispatchableGraphOperator::~DispatchableGraphOperator() {
    if (deleter_) {
        deleter_(&planned_meta_);
    }
}

/* =========================
 * Graph
 * ========================= */

struct Graph::DeviceGraph {
    struct UpdatableTask {
        std::shared_ptr<GraphOperator> op;
        infinirtGraphTaskGroup_t handle;
    };

    infinirtGraph_t graph;
    infinirtGraphExec_t exec;
    infinirtGraphNode_t node;
    std::vector<char> log_buffer;
    std::vector<UpdatableTask> updatable_tasks;

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
            device_graph->launch();
            return;
        }
        for (const auto &op : ops) {
            op->run();
        }
    }
};

Graph::Graph()
    : host_int_arrays_(std::move(staged_host_int_arrays)) {
    staged_host_int_arrays.clear();
}

void Graph::run() const {
    if (segments_.empty()) {
        for (const auto &op : op_list_) {
            op->run();
        }
        return;
    }

    for (const auto &segment : segments_) {
        if (!segment->device_graph) {
            segment->run();
            continue;
        }

        HostIntArrayScope update_scope(&host_int_arrays_, true);
        for (const auto &task : segment->device_graph->updatable_tasks) {
            TaskUpdateHandleScope task_update_scope(task.handle);
            task.op->run();
        }
        segment->device_graph->launch();
    }
}

void Graph::add_operator(std::shared_ptr<GraphOperator> op) {
    op_list_.push_back(op);
}

void Graph::update_host_int_array(
    const Tensor &device_tensor,
    const int32_t *host_values,
    size_t count) {
    INFINICORE_ASSERT(device_tensor);
    INFINICORE_ASSERT(host_values != nullptr || count == 0);

    auto &values = host_int_arrays_[device_tensor->data()];
    values.resize(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<int64_t>(host_values[i]);
    }
}

bool is_task_updating() {
    return task_update_active;
}

bool is_task_group_capturing() {
    return capture_task_group_handle != nullptr;
}

void begin_task_group_capture() {
    INFINICORE_ASSERT(capture_task_group_handle != nullptr);
    INFINICORE_ASSERT(*capture_task_group_handle == nullptr);
    INFINICORE_CHECK_ERROR(
        infinirtGraphTaskGroupBegin(context::getStream()));
}

void end_task_group_capture() {
    INFINICORE_ASSERT(capture_task_group_handle != nullptr);
    INFINICORE_ASSERT(*capture_task_group_handle == nullptr);
    INFINICORE_CHECK_ERROR(infinirtGraphTaskGroupEnd(
        context::getStream(), capture_task_group_handle));
}

void begin_task_update() {
    INFINICORE_ASSERT(task_update_handle != nullptr);
    INFINICORE_CHECK_ERROR(infinirtGraphTaskUpdateBegin(
        context::getStream(), task_update_handle));
}

void end_task_update() {
    INFINICORE_ASSERT(task_update_handle != nullptr);
    INFINICORE_CHECK_ERROR(
        infinirtGraphTaskUpdateEnd(context::getStream()));
}

void stage_task_update_host_int_array(
    const Tensor &device_tensor,
    const int32_t *host_values,
    size_t count) {
    INFINICORE_ASSERT(device_tensor);
    INFINICORE_ASSERT(host_values != nullptr || count == 0);

    auto &values = staged_host_int_arrays[device_tensor->data()];
    values.resize(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<int64_t>(host_values[i]);
    }
}

const std::vector<int64_t> *lookup_task_update_host_int_array(
    const Tensor &tensor) {
    if (current_host_int_arrays == nullptr || !tensor) {
        return nullptr;
    }
    auto it = current_host_int_arrays->find(tensor->data());
    if (it == current_host_int_arrays->end()) {
        return nullptr;
    }
    return &it->second;
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

        device_graph.updatable_tasks.clear();
        HostIntArrayScope capture_scope(&host_int_arrays_, false);
        for (const auto &op : segment->ops) {
            if (!op->requires_task_update()) {
                op->run();
                continue;
            }

            infinirtGraphTaskGroup_t handle = nullptr;
            {
                TaskGroupCaptureScope task_group_scope(&handle);
                op->run();
            }
            INFINICORE_ASSERT(handle != nullptr);
            device_graph.updatable_tasks.push_back({op, handle});
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
