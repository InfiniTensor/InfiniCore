#include "graph_manager.hpp"

#include "../../bridge/infini/rt.hpp"
#include "../utils.hpp"
#include "infinicore/context/context.hpp"

#ifdef USE_INFINIRT_GRAPH
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
    auto current_device = context::getDevice();
    device_graph_->device_type = bridge::infini::rt::translate_to(static_cast<infiniDevice_t>(current_device.getType()));
    device_graph_->device_index = static_cast<int>(current_device.getIndex());
    device_graph_->stream = bridge::infini::rt::translate_to(context::getStream());
    if (device_graph_->device_type == ::infini::rt::Device::Type::kCount) {
        spdlog::warn("InfiniRT graph runtime does not support the current device. Falling back to eager execution.");
        device_graph_.reset();
        return;
    }
    ::infini::rt::set_runtime_device_type(device_graph_->device_type);
    auto set_device_status = bridge::infini::rt::translate(rt_runtime::SetDevice(device_graph_->device_index));
    if (set_device_status != INFINI_STATUS_SUCCESS) {
        spdlog::warn("InfiniRT graph runtime failed to select the current device. Falling back to eager execution.");
        device_graph_.reset();
        return;
    }

    // warmup
    for (size_t iter = 0; iter < 5; ++iter) {
        this->run();
    }
    infinicore::context::syncStream();

    auto begin_status = bridge::infini::rt::translate(rt_runtime::StreamBeginCapture(
        device_graph_->stream,
        rt_runtime::StreamCaptureMode::kStreamCaptureModeRelaxed));
    if (begin_status != INFINI_STATUS_SUCCESS) {
        spdlog::warn("Fail to begin device graph capture.");
        device_graph_.reset();
        return;
    }

    // Run and record
    this->run();

    auto end_status = bridge::infini::rt::translate(rt_runtime::StreamEndCapture(
        device_graph_->stream,
        &device_graph_->graph));
    if (end_status != INFINI_STATUS_SUCCESS) {
        spdlog::warn("Fail to end device graph capture.");
        device_graph_.reset();
        return;
    }

    auto instantiate_status = bridge::infini::rt::translate(rt_runtime::GraphInstantiate(
        &device_graph_->exec,
        device_graph_->graph));
    if (instantiate_status != INFINI_STATUS_SUCCESS) {
        static bool warned_once = false;
        if (!warned_once) {
            warned_once = true;
            spdlog::warn("Fail to instantiate device graph.");
        }
        device_graph_.reset();
        return;
    }
    static bool logged_once = false;
    if (!logged_once) {
        logged_once = true;
        spdlog::info("Using InfiniRT C++ graph runtime API for graph capture and replay.");
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
