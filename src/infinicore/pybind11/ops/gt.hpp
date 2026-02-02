#pragma once
#include "infinicore/ops/gt.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infinicore::ops {
inline void bind_gt(py::module &m) {
    m.def("gt",
          &op::gt,
          py::arg("input"),
          py::arg("other"));

    m.def("gt_",
          &op::gt_,
          py::arg("input"),
          py::arg("other"),
          py::arg("output"));
}
} // namespace infinicore::ops