#pragma once

#include "infinicore/ops/topk.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_topk(py::module &m) {
    m.def("topk",
          &op::topk,
          py::arg("input"),
          py::arg("k"),
          py::arg("dim"),
          py::arg("largest") = true,
          py::arg("sorted") = true,
          R"doc(Returns the k largest elements of the given input tensor along a given dimension.)doc");

    m.def("topk_",
          &op::topk_,
          py::arg("input"),
          py::arg("values"),
          py::arg("indices"),
          py::arg("k"),
          py::arg("dim"),
          py::arg("largest") = true,
          py::arg("sorted") = true,
          R"doc(In-place topk.)doc");
}

} // namespace infinicore::ops