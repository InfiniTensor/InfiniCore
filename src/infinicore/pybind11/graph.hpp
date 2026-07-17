#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore.hpp"

namespace py = pybind11;

namespace infinicore::graph {
inline void bind(py::module_ &m) {
    py::class_<infinicore::graph::Graph,
               std::shared_ptr<infinicore::graph::Graph>>(m, "Graph")
        .def(py::init<>()) // allow construction
        .def("run", &infinicore::graph::Graph::run)
        .def("has_device_exec", &infinicore::graph::Graph::has_device_exec)
        .def("device_segment_count", &infinicore::graph::Graph::device_segment_count)
        .def("device_graph_log", &infinicore::graph::Graph::device_graph_log)
        .def("last_replay_used_device", &infinicore::graph::Graph::last_replay_used_device)
        .def("replay_device_ok", &infinicore::graph::Graph::replay_device_ok)
        .def("replay_op_list_fallback", &infinicore::graph::Graph::replay_op_list_fallback)
        .def("capture_arena_bytes", &infinicore::graph::Graph::capture_arena_bytes)
        .def("capture_arena_blocks", &infinicore::graph::Graph::capture_arena_blocks)
        .def("capture_arena_retained_torch",
             &infinicore::graph::Graph::capture_arena_retained_torch);
}
} // namespace infinicore::graph
