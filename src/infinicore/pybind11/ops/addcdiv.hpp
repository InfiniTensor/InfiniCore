#pragma once

#include "infinicore/ops/addcdiv.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_addcdiv(py::module &m) {
    m.def("addcdiv",
          &op::addcdiv,
          py::arg("input"),
          py::arg("tensor1"),
          py::arg("tensor2"),
          py::arg("value") = 1.0f,
          R"doc(Computes input + value * (tensor1 / tensor2).)doc");

    m.def("addcdiv_",
          &op::addcdiv_,
          py::arg("input"),
          py::arg("tensor1"),
          py::arg("tensor2"),
          py::arg("output"),
          py::arg("value") = 1.0f,
          R"doc(In-place version of addcdiv.)doc");
}

} // namespace infinicore::ops