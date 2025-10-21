#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/op/rearrange.hpp"

namespace py = pybind11;

namespace infinicore::op {

inline void bind_rearrange(py::module &m) {
    m.def("rearrange",
          &op::rearrange,
          py::arg("x"),
          R"doc(Matrix rearrangement of a tensor.)doc");

    m.def("rearrange_",
          &op::rearrange_,
          py::arg("y"),
          py::arg("x"),
          R"doc(In-place tensor rearrangement.)doc");
}

} // namespace infinicore::op
