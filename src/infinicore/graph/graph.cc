#include "graph_manager.hpp"

#include "../utils.hpp"

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

void Graph::run() const {
    for (auto &op : op_list_) {
        op->run();
    }
}

void Graph::add_operator(std::shared_ptr<GraphOperator> op) {
    op_list_.push_back(op);
}

/* =========================
 * GraphManager
 * ========================= */

bool GraphManager::is_recording() const {
    return recording_;
}

void GraphManager::start_recording() {
    recording_ = true;
    graph_ = std::make_shared<Graph>();
}

void GraphManager::add_operator(std::shared_ptr<GraphOperator> op) {
    INFINICORE_ASSERT(recording_);

    graph_->add_operator(op);
}

std::shared_ptr<Graph> GraphManager::stop_recording() {

    recording_ = false;
    return std::exchange(graph_, nullptr);
}

} // namespace infinicore::graph
