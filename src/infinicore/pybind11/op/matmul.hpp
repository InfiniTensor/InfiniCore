#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/op/matmul.hpp"

namespace py = pybind11;

namespace infinicore::op {

inline void bind_matmul(py::module &m) {
    m.def("matmul",
          &op::matmul,
          py::arg("a"),
          py::arg("b"),
          R"doc(Matrix multiplication of two tensors.)doc");

    m.def("matmul_",
          &op::matmul_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place matrix multiplication.)doc");
}

} // namespace infinicore
