#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/index_select.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_index_select(py::module &m) {
    m.def("index_select",
          &op::index_select,
          py::arg("input"),
          py::arg("dim"),
          py::arg("index"),
          R"doc(Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.)doc");

    m.def("index_select_",
          &op::index_select_,
          py::arg("output"),
          py::arg("input"),
          py::arg("dim"),
          py::arg("index"),
          R"doc(In-place index select operation.)doc");
}

} // namespace infinicore::ops
