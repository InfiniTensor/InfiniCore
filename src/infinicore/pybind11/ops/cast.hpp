#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/cast.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_cast(py::module &m) {
    m.def("cast_",
          &op::cast_,
          py::arg("output"),
          py::arg("input"),
          R"doc(Copy a tensor into an output tensor with dtype conversion.)doc");
}

} // namespace infinicore::ops
