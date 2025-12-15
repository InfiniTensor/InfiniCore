#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/logsigmoid.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_logsigmoid(py::module &m) {
    m.def("logsigmoid",
          &op::logsigmoid,
          py::arg("input"),
          R"doc(LogSigmoid activation function.)doc");

    m.def("logsigmoid_",
          &op::logsigmoid_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place LogSigmoid activation function.)doc");
}

} // namespace infinicore::ops

