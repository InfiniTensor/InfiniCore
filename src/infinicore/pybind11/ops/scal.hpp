#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/scal.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_scal(py::module &m) {
    m.def("scal_",
          py::overload_cast<Tensor, Tensor>(&op::scal_),
          py::arg("x"),
          py::arg("alpha"),
          R"doc(In-place BLAS level-1 scal, updating x.)doc");
}

} // namespace infinicore::ops
