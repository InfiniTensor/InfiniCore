#pragma once

#include <memory>
#include <vector>

#include "../tensor.hpp"

namespace infinicore::graph {
// Forward declarations
class GraphManager;

class GraphTensor : public Tensor {
public:
    GraphTensor(const Tensor &);
};

class GraphOperator {

public:
    void run() const;
    ~GraphOperator();

protected:
    using run_schema = void (*)(void *);
    using cleanup_schema = void (*)(void **);
    void *planned_meta_;
    run_schema runner_;
    cleanup_schema deleter_;
};

class Graph {
public:
    Graph() = default;
    ~Graph() = default;

    void run() const;

protected:
    void add_operator(std::shared_ptr<GraphOperator> op);

    std::vector<std::shared_ptr<GraphOperator>> op_list_;

    friend class GraphManager;
};
} // namespace infinicore::graph

#define INFINICORE_GRAPH_OP_CLASS(__OP_NAME__, ...)                        \
    class __OP_NAME__ : public graph::GraphOperator {                      \
    public:                                                                \
        using schema = void (*)(__VA_ARGS__);                              \
        using plan_schema = void *(*)(__VA_ARGS__);                        \
        static common::OpDispatcher<plan_schema> &plan_dispatcher();       \
        static common::OpDispatcher<run_schema> &run_dispatcher();         \
        static common::OpDispatcher<cleanup_schema> &cleanup_dispatcher(); \
        __OP_NAME__(__VA_ARGS__);                                          \
        static void execute(__VA_ARGS__);                                  \
    };

#define INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(__OP_NAME__)                                  \
    common::OpDispatcher<__OP_NAME__::plan_schema> &__OP_NAME__::plan_dispatcher() {       \
        static common::OpDispatcher<__OP_NAME__::plan_schema> dispatcher_;                 \
        return dispatcher_;                                                                \
    }                                                                                      \
    common::OpDispatcher<__OP_NAME__::run_schema> &__OP_NAME__::run_dispatcher() {         \
        static common::OpDispatcher<__OP_NAME__::run_schema> dispatcher_;                  \
        return dispatcher_;                                                                \
    }                                                                                      \
    common::OpDispatcher<__OP_NAME__::cleanup_schema> &__OP_NAME__::cleanup_dispatcher() { \
        static common::OpDispatcher<__OP_NAME__::cleanup_schema> dispatcher_;              \
        return dispatcher_;                                                                \
    }

#define INFINICORE_GRAPH_OP_DISPATCH(__DEVICE_TYPE__, ...)                  \
    planned_meta_ = plan_dispatcher().lookup(__DEVICE_TYPE__)(__VA_ARGS__); \
    runner_ = run_dispatcher().lookup(__DEVICE_TYPE__);                     \
    deleter_ = cleanup_dispatcher().lookup(__DEVICE_TYPE__);

#define INFINICORE_GRAPH_OP_RECORD_OR_RUN(__OP_NAME__, ...) \
    auto op = std::make_shared<__OP_NAME__>(__VA_ARGS__);   \
    if (context::isGraphRecording()) {                      \
        context::addGraphOperator(op);                      \
    } else {                                                \
        op->run();                                          \
    }

#define INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(__OP_NAME__, __PLAN_F__, __RUN_F__, __CLEANUP_F__) \
    static bool registered = []() {                                                               \
        __OP_NAME__::plan_dispatcher().registerAll(__PLAN_F__, false);                            \
        __OP_NAME__::run_dispatcher().registerAll(__RUN_F__, false);                              \
        __OP_NAME__::cleanup_dispatcher().registerAll(__CLEANUP_F__, false);                      \
        return true;                                                                              \
    }();
