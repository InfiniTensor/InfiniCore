#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/argwhere.hpp"
namespace py = pybind11;
namespace infinicore::ops {
inline void bind_argwhere(py::module &m) {
    m.def("argwhere",
          &op::argwhere,
          py::arg("x"),
          R"doc(Argwhere.)doc");
}
}