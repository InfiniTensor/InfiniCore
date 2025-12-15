#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/logical_xor.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_logical_xor(py::module &m) {
    m.def("logical_xor",
          &op::logical_xor,
          py::arg("a"),
          py::arg("b"),
          R"doc(Logical XOR of two tensors.)doc");

    m.def("logical_xor_",
          &op::logical_xor_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place logical XOR of two tensors.)doc");
}

} // namespace infinicore::ops

