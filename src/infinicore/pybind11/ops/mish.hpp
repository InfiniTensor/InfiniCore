#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/mish.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_mish(py::module &m) {
    m.def("mish",
          &op::mish,
          py::arg("input"),
          py::arg("inplace") = false,
          R"doc(Applies the Mish activation function: x * tanh(softplus(x)).)doc");

}

} // namespace infinicore::ops
