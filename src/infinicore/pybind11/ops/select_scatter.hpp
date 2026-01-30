#pragma once
#include "infinicore/ops/select_scatter.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_select_scatter(py::module &m) {

    m.def("select_scatter",
          &op::select_scatter,
          py::arg("input"),
          py::arg("src"),
          py::arg("dim"),
          py::arg("index"));
}

} // namespace infinicore::ops