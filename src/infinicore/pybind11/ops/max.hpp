#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/max_global.hpp"
#include "infinicore/ops/max_reduce.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_max(py::module &m) {
    m.def("max_reduce",
          &op::max_reduce,
          py::arg("input"),
          py::arg("dim"),
          py::arg("keepdim") = false,
          R"doc(Reduces the input tensor along the specified dimension by taking the maximum value.)doc");

    m.def("max_reduce_",
          &op::max_reduce_,
          py::arg("input"),
          py::arg("output"),
          py::arg("indices"),
          py::arg("dim"),
          py::arg("keepdim") = false,
          R"doc(In-place max reduction along the specified dimension.)doc");

    m.def("max_global",
          &op::max_global,
          py::arg("input"),
          R"doc(Reduces the input tensor globally by taking the maximum value across all elements.)doc");

    m.def("max_global_",
          &op::max_global_,
          py::arg("input"),
          py::arg("output"),
          R"doc(In-place global max reduction.)doc");
}

} // namespace infinicore::ops
